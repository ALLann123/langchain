#!/usr/bin/python3
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt
from langgraph.graph import START, END, StateGraph
#from IPython.display import Image, display

class State(TypedDict):
    input:str

def step_1(state:State) -> State:
    print("---Step 1---")
    return state

def step_2(state:State) -> State:
    #lets optically raise a NodeInterrupt if the length of the input is longer than 5 characters
    if len(state['input']) > 5:
        raise NodeInterrupt(f"Received input that is longer that 5 characters: {state['input']}")
    print("---Step 2---")
    return state

def step_3(state:State) -> State:
    print("---Step 3---")
    return state

#build our graph
builder=StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)

#connect the nodes
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

#set up memory
memory=MemorySaver()

#compile our graph
graph=builder.compile(checkpointer=memory)

#view flow
#display(Image(graph.get_graph().draw_mermaid_png()))

#lets run the graph with an input longer than 5 characters to trigger our dynamic breakpoint
initial_input={"input":"hello world"}
thread_config={"configurable": {"thread_id":"1"}}

#run the graph until the first interuption
for event in graph.stream(initial_input, thread_config, stream_mode="values"):
    print(event)

#lets see where execution stoped
state=graph.get_state(thread_config)
print(state.next)

#to continue lets rerun by updating the state
graph.update_state(
    thread_config,
    {"input":"hi"}
)

print("\n[+]Updating State.....\n")

for event in graph.stream(None, thread_config, stream_mode="values"):
    print(event)

"""
ode\AI\Academy LangGraph\module_3> python dynamic_breakpoints.py
{'input': 'hello world'}
---Step 1---                                                            oints.py
{'input': 'hello world'}
{'__interrupt__': (Interrupt(value='Received input that is longer that 5 characters: hello world', resumable=False, ns=None),)}
('step_2',)                                                              characters: hello world', resumable=False, ns=None),)}

[+]Updating State.....

{'input': 'hi'}
---Step 2---
{'input': 'hi'}
---Step 3---
{'input': 'hi'}
"""