#!/usr/bin/python3
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage

#initial state
initial_messages=[AIMessage(content="Hello, how can I assist you?", name="model"),
                  HumanMessage(content="I'm Looking for information on marine biology.", name="Allan")
                  ]

#new message to add
new_message=AIMessage(content="Sure, I can help with that. What specifically interests you?", name="model")

#test
add_messages(initial_messages, new_message)
