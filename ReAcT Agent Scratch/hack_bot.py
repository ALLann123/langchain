#!/usr/bin/python3
from dotenv import load_dotenv
from groq import Groq
import os
import re
import smtplib
from duckduckgo_search import DDGS
from ftplib import FTP, error_perm
import scanless
import json
import requests
import dns.resolver
from langchain_experimental.utilities import PythonREPL
from langchain.tools import Tool

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class Agent:
    def __init__(self, client, system):
        self.client = client
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": self.system})

    def __call__(self, message: str = ""):
        if message:
            self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = self.client.chat.completions.create(
            messages=self.messages,
            model="llama3-70b-8192",
        )
        return completion.choices[0].message.content

# Tool functions
def calculate(operation: str):
    if not re.fullmatch(r"[0-9eE+\-*/. ()]+", operation):
        raise ValueError("Illegal characters in calculation")
    return str(eval(operation))

def duckduckgo_search(query: str) -> str:
    with DDGS() as ddgs:
        return "\n".join(
            f"{res['title']} - {res['href']}\n{res['body']}"
            for i, res in enumerate(ddgs.text(query))
            if i < 3
        )

def send_mail(message: str) -> str:
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login("karisallan237@gmail.com", "none")
            server.sendmail(
                "karisallan237@gmail.com",
                "karisallan237@gmail.com",
                message.encode('utf-8')
            )
        return "Email sent successfully"
    except Exception as e:
        return f"Email failed: {str(e)}"

def get_dns_record(target: str) -> str:
    try:
        return dns.resolver.resolve(target, "A")[0].to_text()
    except Exception as e:
        return f"DNS resolution failed: {str(e)}"

def check_anonymous_login(hostname: str) -> str:
    try:
        with FTP(hostname, timeout=10) as ftp:
            ftp.login()
        return "Anonymous FTP allowed"
    except error_perm:
        return "Anonymous login NOT allowed"
    except Exception as e:
        return f"FTP error: {str(e)}"

def scan_open_ports(target_ip: str) -> str:
    try:
        scanner = scanless.Scanless()
        return json.dumps(scanner.scan(target_ip, scanner="ipfingerprints"), indent=2)
    except Exception as e:
        return f"Scan failed: {str(e)}"

def get_websites_on_server(ip: str) -> str:
    try:
        response = requests.get(
            f"https://api.viewdns.info/reverseip/?host={ip}&apikey={os.getenv('GET_WEBSITES_KEY')}&output=json",
            timeout=10
        )
        data = response.json()
        if domains := data.get("response", {}).get("domains", []):
            return json.dumps([site['name'] for site in domains], indent=2)
        return "No domains found"
    except Exception as e:
        return f"API error: {str(e)}"

def reverse_lookup(target: str) -> str:
    try:
        return dns.resolver.resolve(target, "A")[0].to_text()
    except Exception as e:
        return f"Reverse lookup failed: {str(e)}"

def load_python_repl_tool():
    return Tool(
        name="python_repl",
        func=PythonREPL().run,
        description="A Python REPL for executing code"
    )

# Optimized system prompt
system_prompt = """You are Mr.Robot, a penetration testing AI. Follow this workflow:

1. Thought: Analyze the task
2. Action: Use ONE tool (formatted exactly as shown)
3. PAUSE
4. Observation: Receive tool output
5. Repeat until Answer is ready

TOOLS:
- calculate: <math_expr> (e.g.: "calculate: 5.972e24 * 2")
- get_planet_mass: <planet> (returns mass in kg)
- duckduckgo_search: "<query>" (returns top 3 results)
- send_mail: "<message>" (sends to admin)
- get_dns_record: <domain> (returns IP)
- check_anonymous_login: <ftp_host> (tests FTP access)
- scan_open_ports: <ip> (returns open ports)
- get_websites_on_server: <ip> (finds hosted sites)
- reverse_lookup: <domain> (resolves IP)
- python_repl: <code> (executes Python)

EXAMPLE:
Question: Get IP for ftp.debian.org and check anonymous FTP
Thought: First resolve DNS
Action: get_dns_record: ftp.debian.org
PAUSE
Observation: 130.89.148.14
Thought: Now test FTP access
Action: check_anonymous_login: 130.89.148.14
PAUSE
Observation: Anonymous FTP allowed
Answer: ftp.debian.org (130.89.148.14) allows anonymous access."""

def agent_loop(max_iterations: int, system: str, query: str):
    agent = Agent(client, system)
    tools = {
        "calculate": calculate,
        "get_planet_mass": lambda p: str(get_planet_mass(p)),
        "duckduckgo_search": duckduckgo_search,
        "send_mail": send_mail,
        "get_dns_record": get_dns_record,
        "check_anonymous_login": check_anonymous_login,
        "scan_open_ports": scan_open_ports,
        "get_websites_on_server": get_websites_on_server,
        "reverse_lookup": reverse_lookup,
        "python_repl": load_python_repl_tool().func
    }
    
    next_prompt = query
    for i in range(max_iterations):
        result = agent(next_prompt)
        print(f"Iteration {i+1}: {result}")
        
        if "PAUSE" in result and "Action" in result:
            if match := re.search(r"Action:\s*([a-z_]+):\s*(.+)", result):
                tool, arg = match.groups()
                if tool in tools:
                    try:
                        observation = tools[tool](arg.strip())
                        next_prompt = f"Observation: {observation}"
                    except Exception as e:
                        next_prompt = f"Observation: Tool error - {str(e)}"
                else:
                    next_prompt = "Observation: Invalid tool"
            else:
                next_prompt = "Observation: Malformed action"
            continue
            
        if "Answer" in result:
            return result
    
    return "Maximum iterations reached without answer"

if __name__ == "__main__":
    print("="*60)
    print("PENTEST BOT".center(60))
    print("="*60)
    
    while True:
        try:
            query = input("Own_the_net>> ").strip()
            if query.lower() == "exit":
                print("Fuck Society!!")
                break
                
            result = agent_loop(10, system_prompt, query)
            print(result)
            print("-"*60)
        except KeyboardInterrupt:
            print("\nSession terminated")
            break