#!/usr/bin/python3
import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()

# Load the existing vector store
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "chroma_db")

# Check if the chroma vector store already exists
if os.path.exists(persistent_directory):
    print("[+] Loading existing vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embeddings,
        collection_name="cv_collection",
        collection_metadata={"hnsw:space": "cosine"}
    )
else:
    raise FileNotFoundError(
        f"[-] The directory {persistent_directory} does not exist!!"
    )

# Create a retriever to query the vector store
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.5
    }
)

# Choose either Groq or Ollama - uncomment your preferred option

# Option 1: Groq (fastest option)
model = ChatGroq(
    temperature=0.3,
    model_name="mixtral-8x7b-32768",  # or "llama2-70b-4096"
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Option 2: Ollama (local option)
# model = Ollama(
#     model="llama3",  # or "mistral", "mixtral", etc.
#     temperature=0.3
# )

# Contextualize question prompt
contextualize_q_system_prompt = """Given a conversation about candidate CVs and the latest question, 
reformulate it as a standalone question that focuses on skills, experience, or qualifications 
mentioned in the CVs. Preserve any specific details about technologies, job titles, or education."""

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create a history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt
)

# Answer question prompt for CV analysis
qa_system_prompt = """You are an expert HR assistant analyzing CVs. Use the following CV extracts to answer questions about candidates.
Focus on:
- Skills and technologies mentioned
- Work experience duration and roles
- Education and certifications
- Specific achievements

For missing information, say "This information isn't available in the CV."
Keep answers professional and concise (2-3 sentences max).

Context:
{context}"""

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create a chain to combine documents for question answering
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

# Create a retrieval chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Set up ReAct Agent with Document Store Retriever
react_docstore_prompt = hub.pull("hwchase17/react")

# Tool description for CV queries
tools = [
    Tool(
        name="CV_Database",
        func=lambda input_dict: rag_chain.invoke(input_dict),
        description="""Access information about candidates from their CVs. Input should be questions about:
- Specific skills or technologies
- Work experience in certain roles or industries
- Educational background
- Availability of candidates with specific qualifications"""
    )
]

# Create the ReAct Agent
agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=react_docstore_prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tools,
    handle_parsing_errors=True, 
    verbose=True,
    max_iterations=5
)

chat_history = []

print("="*50)
print("           CV ANALYSIS AGENT")
print("="*50)
print("Ask about candidates' skills, experience, or education")
print("Type 'exit' to quit\n")

while True:
    try:
        query = input("You>> ").strip()
        if query.lower() == "exit":
            break
            
        print("-"*50)
        response = agent_executor.invoke(
            {"input": query, "chat_history": chat_history}
        )
        print(f"\nAI: {response['output']}\n")

        # Update history
        chat_history.extend([
            HumanMessage(content=query),
            AIMessage(content=response["output"])
        ])
        
        # Limit chat history
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]
            
    except Exception as e:
        print(f"\nError: {str(e)}\n")
        continue