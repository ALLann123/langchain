#!/usr/bin/python3
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
import os
from langchain.tools import tool
from kali import connected_kali

#load our environ variables
load_dotenv()

#create the llm: We will use llama from groq
llm = ChatGroq(
    temperature=0,  # Lower temperature for more factual responses
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

@tool
def kali_linux(query:str)->str:
    """Useful for pentesting and CTF's. You can run any kali linux command onthe shell"""
    print("\n[+] Calling OS Kali Linux")
    output=connected_kali(query)
    return output

@tool
def say_hello(name:str)->str:
    """Useful for greeting a user"""
    print("Greeting tool has been called")
    return f"hello, {name} from Own_The_Net"

def main():
    model=llm

    tools=[kali_linux, say_hello]
    agent_executor=create_react_agent(model, tools)

    print("**********"*5)
    print("       Mr.Robot   ")
    print("**********"*5)

    while True:
        user_input=input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("[+]Till next time")
            break

        print("\n Assistant: ", end="")
        
        for chunk in agent_executor.stream(
            {"messages":[HumanMessage(content=user_input)]}
        ): #check if response has agent and messages
            if "agent" in chunk and "messages" in chunk['agent']:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")
        
        print()

if __name__=="__main__":
    main()



"""
 python .\main.py
**************************************************
       Mr.Robot
**************************************************

You: do nmap scan on the domain scanme.nmap.org

 Assistant:
[+] Calling OS Kali Linux

[+] Calling OS Kali Linux
The Nmap scan results for the domain scanme.nmap.org are as follows:

* The host is up and has a latency of 0.27s.
* The following ports are open:
        + 22/tcp (SSH)
        + 80/tcp (HTTP)
        + 9929/tcp (Nping echo)
        + 31337/tcp (Elite)
* The SSH service is running OpenSSH 6.6.1p1 Ubuntu 2ubuntu2.13 on Ubuntu Linux.
* The HTTP service is running Apache httpd 2.4.7 on Ubuntu.
* The Nping echo service is running on port 9929.
* The Elite service is running on port 31337.
* The OS detection suggests that the host is running Linux, with a high probability of it being Ubuntu.
* The traceroute shows the path taken by the packets from the scanning host to the target host.        

Please note that the results may vary depending on the network conditions and the configuration of the target host.


You: whoami     

 Assistant: 
[+] Calling OS Kali Linux
You are currently logged in as the root user.

You: whats the kali linux IP?

 Assistant:
[+] Calling OS Kali Linux

[+] Calling OS Kali Linux
The IP address of the Kali Linux machine is 192.168.1.101.

You: scan the kali for any open ports


 Assistant:
[+] Calling OS Kali Linux
 Assistant:
[+] Calling OS Kali Linux

[+] Calling OS Kali Linux
[+] Calling OS Kali Linux

[+] Calling OS Kali Linux
The Kali Linux scan has revealed that port 22 is open and listening for incoming connections, which is the default port for SSH (Secure Shell) services. This suggests that the Kali Linux system has an SSH server running and is accepting connections from remote clients.
"""