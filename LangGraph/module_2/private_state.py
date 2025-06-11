#!/usr/bin/python3
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

class OverallState(TypedDict):
    foo:int

#used as an intermediate working logic of the graph but not relevant for the overall graph input or output
class PrivateState(TypedDict):
    baz:int

#lets build the nodes
def node_1(state: OverallState)-> PrivateState:
    print("--Node 1--")
    return {"baz": state['foo'] + 1}

def node_2(state: PrivateState) -> OverallState:
    print("---Node 2---")
    return {"foo": state["baz"] + 1}

#build graph
builder=StateGraph(OverallState)
#add the nodes
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)

#logical edge
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)

graph=builder.compile()

#view
display(Image(graph.get_graph().draw_mermaid_png()))
