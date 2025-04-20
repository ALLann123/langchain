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
query="How to Pause and resume cracking?"

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
Wifi Hacking> python .\retriver_action.py

----Relevant Documents----

Document 1:
Kali>aircrack-ng [Handshake_file] –w [wordlist]

Example:

N/B –The speed is dependent on how quick your processor is and if you have any processes that   
are running that can make your computer slower.

Chapter Two
Pause and Resuming Cracking
 Problem:
 Aircrack-ng can take hours and up to days at length when cracking the passwords the problem is 
when you ctrl+c on the keyboard it does not save its progress and when we run it again it       
begins the wordlist again.
Now to bypass the problem and if we quit aircrack-ng and come back in a day or two or a week
or a month we proceed from where we left we will use a tool called John the Ripper.
John The ripper is a tool that can be used to do many things such as cracking but we will use it
differently.
Let’s see how first, we want to display a wordlist using john on the terminal screen:

Source: J:\code\AI\AI\Wifi Hacking\book\Wireless_Pwn.pdf

Document 2:
Using hashcat we are able to pause and resume when we want allowing us to pause and come
back to the cracking.

The arrows point to the network name and the other it’s the password found.
It runs 14 million passwords in less than 2 minutes.

Good Bye Teaser
Practice This:
You have learned how to use crunch, pipe input of one process to another and using
hashcat
Quize: Create a wordlist using crunch and take its output to be used in password cracking on
hashcat.
Spoiler: You do not need to use John in saving progress because hashcat has the capabilities
for these.

Source: J:\code\AI\AI\Wifi Hacking\book\Wireless_Pwn.pdf

Document 3:
Explain:
--restore=upc is the file we stored our previous session and we call it using john and the output
of our previous session is the input into aircrack-ng.

PIPING WORDLIST INTO AIRCRACK-NG
Problem: Wordlists can take a lot of space.
Solution: Solution is we generate a large word list with crunch and pipe it into aircrack-ng
directly.
 Kali>crunch 8 8   # we display the result of these on the screen then will pipe the output of these
command to aircrack-ng

We use the above command to pipe the result from crunch which is displayed on the screen as
input to aircrack-ng.
N/B –Now we are able to use a big wordlist and pipe it directly to aircrack-ng without saving
them to our internal storage.

PIPING WORDLIST TO AIRCRACK-NG WITH PAUSE AND RESUME SUPPORT
We want to be able to:
 Use large wordlists we generate without saving to internal storage.
 Stop cracking and resuming without losing progress.
Algorithm:
1. We generate the wordlist with crunch tool (use any  crunch command you want).
2. We use John the ripper to save and restore our progress on the wordlist
3. Now we want to use aircrack-ng to crack.

Source: J:\code\AI\AI\Wifi Hacking\book\Wireless_Pwn.pdf
"""