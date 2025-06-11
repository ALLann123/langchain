#!/usr/bin/python3
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

class State(TypedDict):
    foo:int

def node_1(state):
    print("--Node 1---")
    return {"foo": state["foo"] +1}

builder= StateGraph(State)
builder.add_node("node_1", node_1)

#logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

#add
graph=builder.compile()

#view
display(Image(graph.get_graph().draw_mermaid_png()))
