#!/usr/bin/python3
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel

#load environment variables
load_dotenv()

#the llm we are using google gemini
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

#define prompt templcat for the movie summary
summary_template=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie critic."),
        ("human", "Provide a brief summary of the movie {movie_name}."),
    ]
)

#define plot analysis step
def analyze_plot(plot):
    plot_template=ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            ("human", "Analyze the plot: {plot}. What are its strengths and weaknesses?"),
        ]
    )
    return plot_template.format_prompt(plot=plot)

#define character analysis step
def analyze_characters(characters):
    character_template=ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            ("human", "Analyze the characters: {characters}. What are their strengths and weaknesses?"),
        ]
    )
    return character_template.format_prompt(characters=characters)

#combine analuses into a final verdict

# Combine analyses into a final verdict
def combine_verdicts(plot_analysis, character_analysis):
    return f"Plot Analysis:\n{plot_analysis}\n\nCharacter Analysis:\n{character_analysis}"

# Simplify branches with LCEL
plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x)) | model | StrOutputParser()
)

character_branch_chain = (
    RunnableLambda(lambda x: analyze_characters(x)) | model | StrOutputParser()
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = (
    summary_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"plot": plot_branch_chain, "characters": character_branch_chain})
    | RunnableLambda(lambda x: combine_verdicts(x["branches"]["plot"], x["branches"]["characters"]))
)

# Run the chain
result = chain.invoke({"movie_name": "Bee Keeper"})

print(result)


"""
\langchain course> python .\parallel_chaining.py
Plot Analysis:
"The Bee Keeper" presents a classic case of style over substance, a common pitfall for action films.  Let's break down its strengths and weaknesses:

**Strengths:**

* **Visually Stylish:** As the synopsis mentions, the film boasts a distinct visual flair. This likely translates to impressive cinematography, set design, and overall aesthetic, creating an engaging viewing experience on a superficial level.
* **Thrilling Action Sequences:**  This is a Statham staple, and "The Bee Keeper" delivers.  The fight choreography and stunts likely provide adrenaline-pumping entertainment, catering to fans of the genre.
* **Intriguing Premise:** The idea of a betrayed operative, code-named "Bee Keeper," seeking revenge against a powerful organization holds potential.  The initial setup likely hooks the audience with the promise of a thrilling conspiracy.

**Weaknesses:**

* **Convoluted and Unsatisfying Narrative:** This is the film's major downfall.  Despite an interesting premise, the plot becomes overly complex and fails to deliver 
a satisfying resolution.  The "conspiracy" likely feels contrived and lacks the necessary depth to be truly engaging.
* **Murky Motivations:**  A crucial element of any revenge story is understanding the protagonist's motivations and the reasons behind the betrayal.  "The Bee Keeper" fails to establish these clearly, leaving the audience disconnected from the character's journey and the overall stakes.  Similarly, the antagonists' motivations remain unclear, reducing their impact and making the conflict feel less compelling.
* **Forced Metaphor:** The beekeeping metaphor, as pointed out, feels tacked on and doesn't organically integrate into the narrative. This attempt to add a layer of depth ultimately falls flat and feels pretentious.
* **Lack of Depth:** While visually impressive and action-packed, the film lacks the necessary emotional depth and character development to resonate with the audience beyond the immediate viewing experience.  This prevents it from being truly memorable.

**Overall:**

"The Bee Keeper" sounds like a typical action flick trying to elevate itself with a more complex plot but failing to execute it effectively. While it provides the expected thrills and visual spectacle, the convoluted narrative, murky motivations, and forced metaphor ultimately hold it back. It caters to fans of Statham's action style but leaves those seeking a more substantial and engaging story feeling unfulfilled.  It's a popcorn thriller best enjoyed with low expectations.

Character Analysis:
The central weakness of the characters in "The Bee Keeper" lies in their lack of depth and unclear motivations.  While Jason Statham's "Bee Keeper" serves as the driving force of the narrative, the audience is given very little insight into *why* he seeks such brutal revenge.  Beyond the basic betrayal, his emotional landscape remains unexplored. This makes it difficult for the audience to truly invest in his journey, despite Statham's committed performance.  He becomes a cipher of vengeance, 
proficient in violence but lacking a compelling internal struggle.

The antagonists suffer a similar fate.  The shadowy organization he targets remains largely faceless and their motives, beyond generic greed or power, are poorly defined.  This lack of clarity diminishes the impact of the conflict.  Without understanding the true nature of the conspiracy or the specific grievances that fuel it, the stakes feel artificially inflated.  The villains become mere obstacles for Statham to overcome, rather than complex adversaries with their own compelling motivations.

Even the metaphorical connection to beekeeping, intended to add a layer of intrigue to the character, feels superficial and ultimately weakens the characterization.  
It's a stylistic flourish that doesn't translate into any meaningful understanding of the "Bee Keeper's" personality or methods.  Instead of enhancing his character, 
it distracts from the core narrative.

A potential strength, though underdeveloped, lies in Statham's inherent charisma and physicality. He convincingly portrays a highly skilled operative, and the action 
sequences showcase his strengths as a performer.  Had the script provided a stronger emotional foundation for the character, these action scenes could have resonated 
on a deeper level.  They might have showcased not just physical prowess but also the internal turmoil driving his quest for revenge.

In short, the characters in "The Bee Keeper" are primarily defined by what they *do* rather than who they *are*.  This reliance on external action over internal motivation ultimately prevents the audience from forming a meaningful connection with them, leaving the film feeling hollow despite its stylish visuals and thrilling action sequences.
"""