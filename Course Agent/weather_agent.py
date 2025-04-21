#!/usr/bin/python3
from dotenv import load_dotenv
import os
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import requests

load_dotenv()

# Initialize the Groq model
#the llm we are using google gemini
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

@tool
def get_weather(city):
    """The tool performs a weather search using the city name"""
    api_key=os.getenv("OPEN_WEATHER")
    
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        
    response = requests.get(url).json()
    return response


prompt_template=hub.pull("hwchase17/react")

tools=[
    get_weather,
]

agent=create_react_agent(model, tools, prompt_template)

agent_executor=AgentExecutor(agent=agent, tools=tools, verbose=True)

print("***********"*6)
print("        Weather BOT")
print("***********"*6)

while True:
    query=input("Customer>> ")
    if query.lower() == "exit":
        print("Shutting down...")
        break
    response=agent_executor.invoke({"input":query})
    print("---------------------------------------------------------------")
    print(f"W Bot: {response['output']}\n")
    print()



"""
  warnings.warn(
******************************************************************
        Weather BOT
******************************************************************
Customer>> Nakuru weather today?


> Entering new AgentExecutor chain...
Question: Nakuru weather today?
Thought: I need to get the weather for Nakuru.
Action: get_weather
Action Input: Nakuru{'coord': {'lon': 36.0667, 'lat': -0.2833}, 'weather': [{'id': 500, 'main': 'Rain', 'description': 'light rain', 'icon': '10d'}], 'base': 'stations', 'main': {'temp': 19.67, 'feels_like': 19.65, 'temp_min': 19.67, 'temp_max': 19.67, 'pressure': 1012, 'humidity': 75, 'sea_level': 1012, 'grnd_level': 819}, 'visibility': 10000, 'wind': {'speed': 1.63, 'deg': 138, 'gust': 2.74}, 'rain': {'1h': 0.36}, 
'clouds': {'all': 82}, 'dt': 1745249082, 'sys': {'country': 'KE', 'sunrise': 1745206265, 'sunset': 1745249844}, 'timezone': 10800, 'id': 184622, 'name': 'Nakuru', 'cod': 200}Thought: The weather in Nakuru is light rain with a temperature of 19.67 degrees Celsius.
Final Answer: The weather in Nakuru is light rain with a temperature of 19.67 degrees Celsius.

> Finished chain.
---------------------------------------------------------------
W Bot: The weather in Nakuru is light rain with a temperature of 19.67 degrees Celsius.


Customer>> Who are you?


> Entering new AgentExecutor chain...
Thought: The question is about my identity. I should explain that I am a large language model.

Final Answer: I am a large language model, trained to be informative and comprehensive. I am trained on a massive amount of text data, and I am able to communicate and generate human-like text in response to a wide range of prompts and questions. For example, I can provide summaries of factual topics or create stories.

> Finished chain.
---------------------------------------------------------------
W Bot: I am a large language model, trained to be informative and comprehensive. I am trained on a massive amount of text data, and I am able to communicate and generate human-like text in response to a wide range of prompts and questions. For example, I can provide summaries of 
factual topics or create stories.


Customer>> Reccomend cloths to where in Juja now


> Entering new AgentExecutor chain...
Question: Reccomend cloths to where in Juja now
Thought: I need to know the current weather in Juja to recommend appropriate clothing.
Action: get_weather
Action Input: Juja{'coord': {'lon': 37.0109, 'lat': -1.1032}, 'weather': [{'id': 803, 'main': 'Clouds', 'description': 'broken clouds', 'icon': '04d'}], 'base': 'stations', 'main': {'temp': 20.89, 'feels_like': 20.89, 'temp_min': 20.89, 'temp_max': 20.89, 'pressure': 1012, 'humidity': 71, 'sea_level': 1012, 'grnd_level': 851}, 'visibility': 10000, 'wind': {'speed': 1.21, 'deg': 51, 'gust': 1.54}, 'clouds': {'all': 59}, 'dt': 1745249045, 'sys': {'country': 'KE', 'sunrise': 1745206080, 'sunset': 1745249576}, 'timezone': 10800, 'id': 184745, 'name': 'Juja', 'cod': 200}Thought: The weather in Juja is 20.89°C with broken clouds, a gentle breeze, and humidity at 71%.

Final Answer: Considering the weather in Juja, I recommend wearing a light, long-sleeved shirt or a t-shirt with a light jacket or sweater. 
Long pants or jeans would be appropriate.  You might also consider carrying a light umbrella in case of a passing shower.

> Finished chain.
---------------------------------------------------------------
W Bot: Considering the weather in Juja, I recommend wearing a light, long-sleeved shirt or a t-shirt with a light jacket or sweater. Long pants or jeans would be appropriate.  You might also consider carrying a light umbrella in case of a passing shower.

"""