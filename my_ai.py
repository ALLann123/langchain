import os
import requests
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def chat():
    print("💬 Groq LLaMA Terminal Chat. Type 'exit' to quit.\n")
    messages = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            print("👋 Exiting...")
            break

        messages.append({"role": "user", "content": user_input})

        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.7
        }

        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload)
            response.raise_for_status()
            data = response.json()
            reply = data["choices"][0]["message"]["content"]
            print(f"\nLLaMA: {reply}\n")
            messages.append({"role": "assistant", "content": reply})
        except requests.exceptions.RequestException as e:
            print("❌ Request failed:", e)
        except Exception as ex:
            print("⚠️ Error:", ex)

if __name__ == "__main__":
    chat()
