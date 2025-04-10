import os
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

# Load the GitHub token
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

if not api_key:
    raise Exception("❌ GitHub token not found. Please set it in .env as GITHUB_TOKEN")

# Setup OpenAI client for GitHub Marketplace endpoint
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=api_key,
)

console = Console()
console.print("[bold cyan]🤖 GPT CLI Chat — powered by GitHub Marketplace GPT-4o[/bold cyan]")
console.print("[green]Type 'exit' to quit[/green]\n")

messages = []

while True:
    try:
        user_input = Prompt.ask("[bold blue]You[/bold blue]")
        if user_input.lower() in ("exit", "quit"):
            console.print("[bold red]👋 Goodbye![/bold red]")
            break

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            messages=messages,
            model="gpt-4o",
            temperature=1,
            max_tokens=4096,
            top_p=1
        )

        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        console.print(f"[bold green]AI:[/bold green] {reply}\n")

    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {str(e)}")
