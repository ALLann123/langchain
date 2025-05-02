#!/usr/bin/python3
import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

#load the exisiting vector store
current_dir=os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(
    current_dir, "db", "chroma_db_with_metadata")

#check if the chroma vector store already exists
if os.path.exists(persistent_directory):
    print("[+]Loading existing vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
else:
    raise FileNotFoundError(
        f"[-]The directory {persistent_directory} does not exist!!"
    )

#create a retriver to query the vector store
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3  # Removed score_threshold as it's not supported
    }
)

# Create the LangChain chat model
model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com",
    temperature=0.3
)

#contextualize question prompt
contextualize_q_system_prompt = (
    "Given a chat and the latest user question "
    "which might influence context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# create a history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt
)

# Answer question prompt
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, respond with "
    "I do not know. Use three sentences maximum and keep the answer "
    "concise. "
    "\n\n"
    "{context}"
)

# create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# create a chain to combine documents for question answering
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

# create a retrieval chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#set up ReAct Agent with Document Store Retriever
react_docstore_prompt=hub.pull("hwchase17/react")

tools=[
    Tool(
        name="Answer Question",
        func=lambda input, chat_history=None, **kwargs: rag_chain.invoke(
            {"input": input, "chat_history": chat_history or []}
        ),
        description="useful for when you need to answer questions about the context",
    )
]

#create the ReAct Agent
agent=create_react_agent(
    llm=model,
    tools=tools,
    prompt=react_docstore_prompt,
)

agent_executor=AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tools,
    handle_parsing_errors=True, 
    verbose=False,
    max_iterations=5
)

chat_history=[]

print("******"*6)
print("         AGENT RAG")
print("******"*6)
print("type 'exit' to close!!")
while True:
    print()
    query=input("You>> ")
    print("---------------------------------------------------------")
    if query.lower() == "exit":
        break
    try:
        response=agent_executor.invoke(
            {"input":query, "chat_history":chat_history}
        )
        print(f"AI: {response['output']}")

        #update history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=response["output"]))
        
        # Limit chat history
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
    except Exception as e:
        print(f"Error: {str(e)}")
        continue

"""
************************************
         AGENT RAG
************************************
type 'exit' to close!!

You>> who is Eliot?
---------------------------------------------------------


> Entering new AgentExecutor chain...
Thought: I need more context to determine who "Eliot" refers to, as it could be a person, character, or something else. Let me gather more information.  
Action: Answer Question  
Action Input: Who is Eliot?  
{'input': 'Who is Eliot?  \n', 'chat_history': [], 'context': [Document(id='77b8c8e8-2347-4471-92fa-0e772a582d7f', metadata={'source': 'season_two.txt'}, page_content="The revolution may have started, but Elliot's war with himself is just beginning."), Document(id='f7f50784-73db-420f-9c88-89af60c3f0e3', metadata={'source': 'season_four.txt'}, page_content='But itâ€™s not real. And in the most selfless act, Elliot chooses to abandon the illusion so the "real" Elliotâ€”the one long hidden deep insideâ€”can return.\nThe revolution wasnâ€™t about toppling corporations. It was about finding yourself in a world thatâ€™s always trying 
to erase you.'), Document(id='20750e9b-61a4-490e-935e-67b0ba1fc122', metadata={'source': 'season_four.txt'}, page_content='Season 4\nThe end begins.\nElliot and Mr. Robot are no longer enemiesâ€”they are allies. The split identities now work together, targeting the Deus Group, the architects of global control. With help from Darlene, they pull off a digital heist for the ages: wiping out the accounts of the rich and powerful.\nBut Whiterose is still ahead. Her machine, housed in the Washington Township power plant, is nearing activation. Angelaâ€™s death looms heavy over everyone. And Elliot, digging deeper into his past, uncovers a horrifying truth: his mind split long ago, not just into Mr. Robotâ€”but into multiple personalities to shield him from childhood trauma.\nThe final act is mind-bending. Elliot discovers an entire alternate realityâ€”perhaps created by the machine, or perhaps by his own fractured psyche. In this other world, his loved ones are happy. His parents are alive. He never became a hacker.')], 'answer': 'Elliot Alderson is the main character of *Mr. Robot*, a cybersecurity engineer and hacker with dissociative identity disorder. He struggles with mental health issues, including anxiety, depression, and a fractured psyche, which manifests as multiple personalities, including Mr. Robot. His journey revolves around battling corporate control, uncovering personal truths, and reconciling with his own identity.'}Final Answer: Eliot refers to Elliot Alderson, the main character of *Mr. Robot*. He is a cybersecurity engineer and hacker who struggles with dissociative identity disorder and mental health issues. His journey involves battling corporate control, uncovering personal truths, and reconciling with 
his fractured psyche, which manifests as multiple personalities, including Mr. Robot.

> Finished chain.
AI: Eliot refers to Elliot Alderson, the main character of *Mr. Robot*. He is a cybersecurity engineer and hacker who struggles with dissociative identity disorder and mental health issues. His journey involves battling corporate control, uncovering personal truths, and reconciling with his fractured psyche, which manifests as multiple personalities, including Mr. Robot.

You>>
"""


"""
AGENT RAG verbose off:
************************************
         AGENT RAG
************************************
type 'exit' to close!!

You>> Who is Elliot's sister and who is his best friend?
---------------------------------------------------------
AI: Elliot's sister is Darlene, and his best friend is Angela Moss.

You>> why does Mr.Robot exist in Elliot's life?
---------------------------------------------------------
AI: Mr. Robot exists in Elliot's life as a figment of his imagination, created by his fractured psyche to cope with childhood trauma. He represents a projection of Elliot's deceased father and serves as a mechanism to shield Elliot from painful memories while driving his revolutionary actions.

You>> Elliot's Hacker group name and what is there agender?
---------------------------------------------------------
AI: Elliot's hacker group is called fsociety. Their agenda is to dismantle corporate control and inequality by targeting E Corp ("Evil Corp") and erasing global debt through the "Five/Nine" hack.

You>> did the hack work?
---------------------------------------------------------
AI: The hack initially worked but caused unintended widespread suffering, prompting Elliot to try to fix the consequences.

You>> exit
"""
