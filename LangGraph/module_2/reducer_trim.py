#!/usr/bin/python3
from langchain_core.messages import RemoveMessage
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from pprint import pprint

load_dotenv()

load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model using the GitHub Marketplace endpoint
llm= ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

#create nodes
def filter_messages(state:MessagesState):
    #delete all but the 2 most recent messages
    delete_messages=[RemoveMessage(id=m.id) for m in state["messages"][:2]]
    return {"messages": delete_messages}

def chat_model_node(state:MessagesState):
    return {"messages": llm.invoke(state["messages"])}

#build graph
builder=StateGraph(MessagesState)
builder.add_node("filter", filter_messages)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "filter")
builder.add_edge("filter", "chat_model")
builder.add_edge("chat_model", END)

graph=builder.compile()


#now add our prompts/messages
# Message list with a preamble
messages = [AIMessage("Hi.", name="Bot", id="1")]
messages.append(HumanMessage("Hi.", name="Lance", id="2"))
messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
messages.append(HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance", id="4"))

#invoke 
output=graph.invoke({'messages':messages})
for m in output['messages']:
    m.pretty_print()


"""
python reducer_trim.perify failed: certificate has expired (_ssl.c:992)')))"))y
================================== Ai Message ==================================                                                                8896-01f6ab074f68; trace=02f41c1e-5140-4ee6-8896-01f6ab074f68,id=70c6449f-e790-4422-9522-5a635489347e; trace=02
Name: Bot                                                               37

So you said you were researching ocean mammals?
================================ Human Message =================================
Name: Lance

Yes, I know about whales. But what others should I learn about?
================================== Ai Message ==================================

That's awesome, Lance! Whales are fascinating, but there are many other 
incredible ocean mammals worth learning about. Here's a list to get you 
started:

### **Dolphins**
- **Bottlenose Dolphins**: The most well-known species, super intelligent and playful.
- **Orcas (Killer Whales)**: Technically the largest member of the dolphin family, known for their complex social structures and hunting techniques.
- **Spinner Dolphins**: Get their name from their acrobatic spins out of the water.

### **Porpoises**
- Often confused with dolphins, but they’re smaller and have rounded teeth and shorter beaks.
- **Vaquita**: The most endangered marine mammal, found only in the northern part of the Gulf of California.

### **Seals**
- Seals thrive in both polar and temperate waters. Some notable ones:   
  - **Harbor Seal**: Found along coastlines, they’re very adaptable.    
  - **Leopard Seal**: An Antarctic predator with a taste for penguins!  

### **Sea Lions**
- Related to seals, but they have external ear flaps and can "walk" on land using their flippers.
- **California Sea Lions**: The playful and noisy sea lions often seen doing tricks in marine parks.

### **Walruses**
- Iconic Arctic mammals with long tusks, used for breaking ice and defense. They’re also social creatures that gather in large groups.

### **Manatees and Dugongs**
- Known as "sea cows," they’re gentle giants that graze on seagrass.    
  - **Manatees**: Found in warm coastal waters in the Americas and Africa.
  - **Dugongs**: Found in coastal waters of the Indo-Pacific, related to manatees but have a slightly different tail shape.

### **Sea Otters**
- Not technically ocean mammals like whales, but they’re marine-adapted 
mammals that spend much of their time in water. They’re known for using 
rocks as tools to crack open shellfish.

### **Polar Bears**
- Technically marine mammals because they spend most of their time on Arctic sea ice, hunting for seals.

### **Sperm Whales & Beaked Whales**
- While you mentioned whales, don’t forget deep-diving specialists like 
sperm whales and lesser-known beaked whales.

### Why Learn About Them?
Ocean mammals are critical to marine ecosystems and are indicators of ocean health. Many face threats like climate change, pollution, and bycatch. Learning about their behaviors, diets, habitats, and conservation challenges can help inspire efforts to protect them.

Which species do you think you'd like to learn more about next, Lance?  
"""