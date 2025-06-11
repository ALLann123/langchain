#!/usr/bin/python3
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage


#message list
messages=[AIMessage("Hi.", name="Bot", id=1)]
messages.append(HumanMessage("Hi", name="Lance", id=2))
messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id=3))
messages.append(HumanMessage("Yes, I know abou whales. But what others shold I learn?",name="Lance", id=4))

#isolate messages to delete
delete_messages=[RemoveMessage(id=m.id) for m in messages[:2]]

#lets do the delete
add_messages(messages, delete_messages)