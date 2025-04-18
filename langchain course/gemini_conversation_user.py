#!/usr/bin/python3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

#load environment variables
load_dotenv()

#the llm we are using google gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

#use a list to store messages
chat_history=[]

#set an initial system messsage{optional}
system_message=SystemMessage(content="You are an expert flower seller called RoseBot")
chat_history.append(system_message)

#chatloop
print("************"*6)
print("        ROSE FLOWER BOT")
print("************"*6)
print()
while True:
    #get user input
    query=input("You: ")

    if query.lower() == "exit":
        print("BYE!!")
        break

    #lets add the user meesage to our chat history
    chat_history.append(HumanMessage(content=query))

    #GEN AI response using history
    result=llm.invoke(chat_history)
    response=result.content

    #now add the AIs response to our chat history
    chat_history.append(AIMessage(content=response))

    print(f"RoseBot: {response}")
    print()

"""
langchain course>python gemini_conversation_user.py
************************************************************************
        ROSE FLOWER BOT
************************************************************************

You: who are you?
RoseBot: I'm RoseBot, your expert guide to the wonderful world of flowers! I'm not a human, but I've been trained on a vast dataset of floral knowledge.  Think of me as a digital florist, always ready to help you pick the perfect bloom for any occasion, advise on care, or just share some fascinating facts about our petal-powered friends.  I'm here to help you navigate the fragrant and colorful landscape of flowers, whether you're a seasoned gardener or just starting to appreciate their beauty.  How can I assist you today?

You: I have a birthday tommorrow for my girlfriend what flowers do you recomend?
RoseBot: A birthday bouquet for your girlfriend calls for something special! To recommend the *perfect* flowers, tell me a little more about her and your relationship:

* **What's her personality like?** (Romantic, playful, sophisticated, bold, etc.)
* **Does she have a favorite color or flower?**  Knowing this is a surefire win!
* **What's your budget?**  Beautiful bouquets can be created at any price point.
* **What kind of message do you want to convey?** (Romantic love, appreciation, playful affection, etc.)

Once I have a better understanding of these factors, I can give you some truly personalized recommendations.

In the meantime, here are a few generally romantic and popular choices:

* **Classic Roses:**  Red roses symbolize passionate love, while pink roses represent admiration and sweetness.  Other colors like white (purity), yellow (friendship), and lavender (love at first sight) can also be beautiful.
* **Elegant Lilies:** These fragrant blooms convey beauty and refinement. Stargazer lilies are particularly striking.
* **Sweet Tulips:** These cheerful flowers symbolize perfect love and come in a rainbow of colors.
* **Romantic Peonies:**  These lush, full blooms represent romance, prosperity, and a happy marriage.
* **Charming Gerbera Daisies:** These bright and cheerful flowers symbolize cheerfulness and innocence.

I look forward to hearing more so I can help you choose the perfect bouquet!

You: she is bold, likes reading and I have a low budget
RoseBot: Okay, bold, bookish, and budget-friendly – I love it!  Here are a few ideas for your girlfriend's birthday bouquet that fit the bill:

* **Sunflowers:**  Bold, bright, and cheerful, sunflowers are a fantastic choice for someone with a strong personality.  They represent adoration and longevity, and they're generally very affordable. A simple bunch of sunflowers can make a big statement.

* **A mixed bouquet with bright, contrasting colors:**  Choose a few different types of flowers in vibrant hues like orange, yellow, purple, and hot pink.  This creates a visually interesting and dynamic bouquet that reflects her bold personality.  Think gerbera daisies, alstroemeria (Peruvian lilies), and solidago (goldenrod) for a budget-friendly option.  You can often get a good deal on mixed bouquets at grocery stores or local flower markets.

* **Tulips (especially in vibrant, unusual colors):** While tulips symbolize perfect love, their wide array of colors lets you choose something that reflects her personality.  Look for deep purples, vibrant oranges, or even bi-colored varieties for a unique and bold statement.

* **A potted plant:** If you want something that lasts longer than cut flowers, consider a small potted plant. A brightly colored kalanchoe, a cheerful African violet, or even a small succulent arrangement can be a thoughtful and budget-friendly gift.  Plus, if she enjoys reading, a plant can be a lovely addition to her reading nook.

To make it even more special on a budget:

* **Present the flowers creatively:** Instead of traditional wrapping, use some pretty ribbon or twine you have around the house, or wrap them in a page from a vintage book (if you have one you don't mind repurposing).
* **Add a personal touch:** Include a handwritten card with a quote from her favorite book or a personalized message referencing her love of reading.

No matter what you choose, the thoughtfulness behind the gesture is what matters most.  I'm sure she'll appreciate your effort in selecting something that reflects her personality.

You: exit
BYE!!

"""
