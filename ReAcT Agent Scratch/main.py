#!/usr/bin/python3
from dotenv import load_dotenv
from groq import Groq
import os
import re

# load the api keys
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


class Agent:
    def __init__(self, client, system):
        self.client = client
        self.system = system
        # this is going to allow our agent to store conversation in memory
        self.messages = []
        # lets add the system message to our chat history
        if self.system:                              # ← fix: add when system prompt IS provided
            self.messages.append({"role": "system", "content": self.system})

    # this is executed whenever we call the agent
    def __call__(self, message: str = ""):
        # add the users message to our chat history
        if message:
            self.messages.append({"role": "user", "content": message})
        # now lets send to the agent and call the execute function
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = self.client.chat.completions.create(   # ← fix: use self.client
            # pass in our message history to add additional context to our agent
            messages=self.messages,
            model="llama3-70b-8192",
        )
        return completion.choices[0].message.content


# lets write the system prompt
system_prompt = system_prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

get_planet_mass:
e.g. get_planet_mass: Earth
returns weight of the planet in kg

Example session:

Question: What is the mass of Earth times 2?
Thought: I need to find the mass of Earth
Action: get_planet_mass: Earth
PAUSE 

You will be called again with this:

Observation: 5.972e24

Thought: I need to multiply this by 2
Action: calculate: 5.972e24 * 2
PAUSE

You will be called again with this: 

Observation: 1,1944×10e25

If you have the answer, output it as the Answer.

Answer: The mass of Earth times 2 is 1,1944×10e25.

Now it's your turn:
""".strip()


# tools for the agent
def calculate(operation):
    # basic whitelist for safety
    if not re.fullmatch(r"[0-9eE+\-*/. ()]+", operation):
        raise ValueError("Illegal characters in calculation")
    return eval(operation)


# in the second one lets use our internet search placed in a function
def get_planet_mass(planet: str) -> float | None:
    match planet.lower():
        case "earth":
            return 5.972e24
        case "mars":
            return 6.39e23
        case "jupiter":
            return 1.898e27
        case "saturn":
            return 5.683e26
        case "uranus":
            return 8.681e25
        case "neptune":
            return 1.024e26
        case "mercury":
            return 3.285e23
        case "venus":
            return 4.867e24
        case _:
            return None


def agent_loop(max_iterations, system, query):
    agent = Agent(client, system)          # ← fix: pass caller‑supplied system
    tools = ["calculate", "get_planet_mass"]
    next_prompt = query
    i = 0
    while i < max_iterations:
        i += 1
        result = agent(next_prompt)
        print(result)
        if "PAUSE" in result and "Action" in result:
            action = re.findall(r"Action:\s*([a-z_]+)\s*:\s*(.+)", result, re.IGNORECASE)
            if not action:
                next_prompt = "Observation: ACTION_NOT_PARSED"
                continue

            chosen_tool = action[0][0]
            arg = action[0][1]

            if chosen_tool in tools:
                try:
                    result_tool = eval(f"{chosen_tool}('{arg}')")
                except Exception as e:
                    result_tool = f"ERROR: {e}"
                next_prompt = f"Observation: {result_tool}"
            else:
                next_prompt = "Observation: TOOL_NOT_FOUND"
            print(next_prompt)
            continue

        if "Answer" in result:
            break


agent_loop(
    max_iterations=10,
    system=system_prompt,
    query="What is the mass of the earth plus mass of mercury and add all of plus 5?"
)


"""
ReAcT Agent Scratch> python .\main.py
Thought: I need to find the mass of Earth and Mercury and add them together, then add 5 to the result.
Thought: I need to find the mass of Earth first.
Action: get_planet_mass: Earth
PAUSE
Observation: 5.972e+24
Thought: Now I need to find the mass of Mercury.
Action: get_planet_mass: Mercury
PAUSE
Observation: 3.285e+23
Thought: Now I have the masses of both planets, I need to add them together.
Action: calculate: 5.972e24 + 3.285e23
PAUSE
Observation: 6.300500000000001e+24
Thought: Now I have the sum of the masses, I need to add 5 to it.
Action: calculate: 6.300500000000001e+24 + 5
PAUSE
Observation: 6.300500000000001e+24
Thought: The result is still very large, but I can output the final answer.
Answer: The mass of the earth plus mass of mercury and add all of plus 5 is 6.300500000000001e+24.
"""
