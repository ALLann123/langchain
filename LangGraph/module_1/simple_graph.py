#!/usr/bin/python3
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
import random
from typing import Literal

#state-serves as the input schema for all nodes
class State(TypedDict):
    graph_state:str

#Nodes- is just a python function(call tools, retriever) that overrides the state
def node_1(state):
    print("--Node 1--")
    #we override the state to say I am
    return {"graph_state": state['graph_state'] + "I am "}

def node_2(state):
    print("--Node 2--")
    return {"graph_state":state['graph_state']+ "happy"}

def node_3(state):
    print("--Node 3--")
    return {"graph_state":state['graph_state']+ "sad"}

#Edges-connects the nodes. Conditional edges used if we want to make a decision between routing two nodes based upon some logic
#-----Normal Edges used if we want to connect two nodes i.e Node_1, to Node_2
def decide_mood(state)->Literal["node_2", "node_3"]:
    user_input=state['graph_state']

    #lets do a 50/50 split between nodes 2,3
    if random.random() < 0.5:
        #50% of the time we return Node 2
        return "node_2"
    
    #also 50 percent of the time we return Node 3
    return "node_3"


#lets build our graph
builder=StateGraph(State)

#lets add our nodes to the graph
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

#logic/conditions
#start-sends the user input to indicate to the graph where to start
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
#END-represents the termination point of our graph
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

#lets compile our graph
app=builder.compile()

#view the image represenation as a graph
#display(Image(app.get_graph().draw_mermaid_png))

#now lets invoke the state of the graph
result=app.invoke({"graph_state":"Hi, this is Allan "})
print(result)