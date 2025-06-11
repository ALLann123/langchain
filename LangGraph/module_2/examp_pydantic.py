#!/usr/bin/python3
import random
from langgraph.graph import StateGraph, START, END
#from IPython.display import Image, display
from typing import Literal
from pydantic import BaseModel, field_validator, ValidationError

class PydanticState(BaseModel):
    name:str
    mood:str #"happy" or "sad"

    @field_validator('mood')
    @classmethod
    def validate_mood(cls, value):
        #Ensure the mood is either "happy" or "sad"
        if value not in ["happy", "sad"]:
            raise ValueError("Each Mood must be either  'Happy' or 'sad' " )
        return value

#lets create our nodes
def node_1(state):
    print("---Node 1---")
    return {"name":state['name'] + " is ..."}

def node_2(state):
    print("--Node 2--")
    return {"mood":"happy"}

def node_3(state):
    print("--Node 3--")
    return {"mood":"sad"}

def decide_mood(state)->Literal["node_2", "node_3"]:
    if random.random()<0.5:
        return "node_2"
    
    return "node_3"

#lets build the graph
builder=StateGraph(PydanticState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

#now logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

#Add
graph=builder.compile()

"""
#view
display(Image(graph.get_graph().draw_mermaid_png()))
"""
result=graph.invoke({"name":"Allan "})
print(result)
