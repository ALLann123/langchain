#!/usr/bin/python3
import os
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma

#define the persistent directory
current_dir=os.path.dirname(os.path.abspath(__file__))
persistent_directory=os.path.join(current_dir, "db", "chroma_db")

#define embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Specify device
    encode_kwargs={'normalize_embeddings': True}  # Helps with similarity
)

#load existing vector store with the embedding function
db=Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

#define the user's question
query="What is FSociety?"

#retruve relevant documents based on query
retriever = db.as_retriever(
    search_type="similarity",  # Changed to basic similarity search
    search_kwargs={"k": 3}     # Removed score_threshold parameter
)

relevant_docs=retriever.invoke(query)

print("\n----Relevant Documents----\n")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")




"""
RAG> python .\query_no_llm.py  

----Relevant Documents----

Document 1:
The second season ended with more questions than answers. Tyrell Wellick (Martin WallstrÃ¶m), a former high-ranking E Corp executive turned fsociety convert, initiates Stage 2 of a still mysterious plan. Elliot lays bleeding in front of Wellick, the result of him trying to put a stop to the plan. The last remaining members of the original fsociety have been approached by a known Dark Army assassin. And the FBI have Darlene, showing her that they've put the pieces together.

Source: J:\code\AI\AI\langchain course\RAG\document\mrrobot.txt

Document 2:
Elliot follows Mr. Robot, the leader of fsociety, a small underground hacker group with eyes on taking down E Corp, a soul-sucking conglomerate that owns a majority of the world's credit debt. The elaborate plan had the team break into the facility where the paper records were kept and destroy them, hacking in and deleting the digital data. Their plan succeeds, thanks largely to the help of Whiterose (B.D. Wong), the leader of the Dark Army, a dangerous Chinese hacker collective with connections worldwide.

Source: J:\code\AI\AI\langchain course\RAG\document\mrrobot.txt

Document 3:
Although the plan went off without a hitch, the second season begins with the world in a dark place. With the records gone, the debt that was 
paid has been reinstated, and all of the world that was relying on E Corps credit is now saddled with a minuscule daily maximum withdrawal. The FBI cracks down on fsociety, the Dark Army are looking to tie up all of their loose ends, and E Corp is investigating those responsible for 
the attacks. On top of this, Elliot is in prison and doesn't seem all that interested in following through with what he has already started. Even when he gets out, his mental state is such that he doesn't even know what he's done. Mr. Robot was in charge, and now Elliot must figure out exactly what he did while his sister, Darlene (Carly Chaikin), struggles to capitalize as fsociety's members get picked off one by one.    

Source: J:\code\AI\AI\langchain course\RAG\document\mrrobot.txt


"""