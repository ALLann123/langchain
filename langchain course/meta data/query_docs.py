#!/usr/bin/python3
import os
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma

#define persistent directory
current_dir=os.path.dirname(os.path.abspath(__file__))
db_dir=os.path.join(current_dir,"db")
persistent_directory=os.path.join(db_dir, "chroma_db_with_metadata")

#define the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Specify device
    encode_kwargs={'normalize_embeddings': True}  # Helps with similarity
)

#load the existing vector store
db=Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

#define the users question
query="Who controls FSociety?"

#Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity",  # Changed to basic similarity search
    search_kwargs={"k": 3}     # Removed score_threshold parameter
)

relevant_docs=retriever.invoke(query)

#display the relevant results with metadata
print("\n---Relevant Documents----")
for i, doc in enumerate(relevant_docs,1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source:{doc.metadata['source']}\n")


"""
a data> python .\query_docs.py

---Relevant Documents----
Document 1:
In a final revelation, Elliot hacks the Deus Groupâ€”Whiteroseâ€™s secret cabalâ€”and uncovers the full scale of their power. He sets his sights on them, fully aware 
that the battle is no longer just digital.

Itâ€™s psychological. Philosophical. Personal.

Source:season_three.txt

Document 2:
Season 3

Elliot is out of prisonâ€”and determined to undo what he started.

The Five/Nine hack didnâ€™t save the world. It plunged it into chaos. People are suffering, and he wants to make it right. But the world heâ€™s trying to fix is being manipulated by forces far more powerful than he imagined. Whiterose, through her shell company, is orchestrating events with frightening precision. She has plans involving a machineâ€”something tied to parallel realities.

Mr. Robot and Elliot now clash, each trying to take control of the same body. One wants destruction. The other, redemption. Elliot works at E Corp, hoping to rebuild, while fsociety fractures.

Through it all, relationships fray. Angela descends into madness, fully devoted to Whiteroseâ€™s vision. Darlene is caught between the FBI and loyalty to Elliot. The 
noose tightens.

Source:season_three.txt

Document 3:
Season 1

In the shadows of New York City, a brilliant but socially withdrawn cybersecurity engineer named Elliot Alderson lives a double life. By day, he protects networks at 
Allsafe Cybersecurity; by night, heâ€™s a vigilante hacker exposing pedophiles, cheating spouses, and corporate corruption. Tormented by social anxiety, depression, and a morphine addiction, Elliot prefers code over conversation.

Everything changes when he meets a mysterious anarchist known only as Mr. Robot, who recruits him into an underground hacking collective called fsociety. Their mission: to bring down one of the worldâ€™s most powerful conglomeratesâ€”E Corp, which Elliot personally calls "Evil Corp." As fsocietyâ€™s planâ€”"Five/Nine"â€”unfolds, Elliot battles internal demons and external enemies.

But in the most shocking twist, Elliot learns that Mr. Robot isnâ€™t real. Heâ€™s a figment of Elliotâ€™s imaginationâ€”his mindâ€™s projection of his dead father.   

Source:season_one.txt
"""