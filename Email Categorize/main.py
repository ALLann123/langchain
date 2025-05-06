#!/usr/bin/python3
from langchain_groq import ChatGroq # type: ignore
from dotenv import load_dotenv # type: ignore
import os
from pprint import pprint
from langchain.prompts import PromptTemplate # type: ignore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

from langchain.schema import Document # type: ignore
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import List

load_dotenv()
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)


#log output for debugging or documentation purposes
def write_markdown_file(content, filename):
    """Writes the given content as markdown file to the local directory.
    Args:
        content:The string content to write to the file
        filename:The filename to save the file as
    """

    with open(f"{filename}.md", "w") as f:
        f.write(content)

#prompt to categorize email
prompt=PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an Email Categorizer Agent. You are a master at understanding what a customer wants when they write an email and are able to categozize it in a useful way.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
     Conduct a comprehensive analysis of the email provided and categorize it into one of the following categories:
        price_enquiry - used when someone is asking for information about pricing \
        customer_complaint - used when someone is complaining about something \
        product_enquiry - used when someone is asking for information about a product feature, benefit, or service but not about pricing \
        customer_feedback - used when someone is giving feedback about a product \
        off_topic - when it doesn't relate to any other category.
      Output a single category only from the types ('price_enquiry', 'customer_complaint', 'product_enquiry', 'customer_feedback', 'off_topic') \
      
      e.g.:
            'price_enquiry'\
    EMAIL CONTENT:\n\n {initial_email} \n\n
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["initial_email"],
)

#lets build our chain and trigger our model
email_category_generator=prompt | llm | StrOutputParser()

#test the chain above
EMAIL="""HI there, \n
I am emailing to say that I had a wonderful stay at your resort last week.\n
I really appreciate what yor staff did

Thanks, 
Paul
"""

#pass the email to our chain and invoke it
#print(f"Email:\n {EMAIL}")
#print("\n[+]Categorizing Email.......\n")

email_category=email_category_generator.invoke({"initial_email":EMAIL})
#print(f"AI: {email_category}")


#now the research prompt to decide whether to draft the email response or gather more information
research_router_prompt=PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert at reading the initial email and routing web search
    or directly to a draft email
    Use the followinf criteria to decide how to route the email:
    If the initial email only requires a simple reason:
    -Choose 'draft_email' for questions you can easily answer,
    including prompt engineering and adversarial attacks.
    -If the emailis just saying thank you, etc., choose 'draft_email'

    Otherwise, use 'research_info.'

    Given a binary choice 'research_info' or 'draft_emaipl' based on the question.
    Return a JSON with a single key 'router_decision' and no preamble or explanation.
    Use both the inital email and email category to make your decision.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Email to route INITIAL_EMAIL: {initial_email}
    EMAIL_CATEGORY: {email_category}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["initial_email", "email_category"]
)


#connect the research prompt to LLM
research_router=research_router_prompt | llm | JsonOutputParser()

#testing the router
#print("**************************************************************************************")
research_result=research_router.invoke({"initial_email": EMAIL, "email_category":email_category})
#print(f"RESEARCH CATEGORY>> {research_result}")

#--Generating Keywords for search
search_keyword_prompt=PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
      You are a master at working out the best keywords for a web search
      to find the most relecant information for the customer.

      Given the INITIAL_EMAIL and EMAIL_CATEGORY, work out the best 
      keywords that will find the most relevant information to help write
      the final email.

      Return a JSON with a single key 'keywords' containing no more than
      3 keywords, and no preamble or explanation.

      <|eot_id|><|start_header_id|>user<|end_header_id|>
      INITIAL_EMAIL:{initial_email}
      EMAIL_CATEGORY:{email_category}
      <|eot_id|><|start_header_id|>assistant<|end_header_id>
      """,
      input_variables=["initial_email", "email_category"],     
)

#create seearch keyword chain
search_keyword_chain=search_keyword_prompt | llm |JsonOutputParser()
#test it
search_result=search_keyword_chain.invoke({"initial_email":EMAIL, "email_category":email_category})
#print("***********************************************************\n")
#print("AI", search_result)

#---Draft the email
draft_writer_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are the Email Writer Agent. Take the INITIAL_EMAIL below from a human that has emailed our company email address, the email_category \
            that the categorizer agent gave it, and the research from the research agent, and \
            write a helpful email in a thoughtful and friendly way.

            If the customer email is 'off_topic' then ask them questions to get more information.
            If the customer email is 'customer_complaint' then try to assure we value them and that we are addressing their issues.
            If the customer email is 'customer_feedback' then thank them and acknowledge their feedback positively.
            If the customer email is 'product_enquiry' then try to give them the info the researcher provided in a succinct and friendly way.
            If the customer email is 'price_enquiry' then try to give the pricing info they requested.

            You never make up information that hasn't been provided by the research_info or in the initial_email.
            Always sign off the emails in an appropriate manner and from Allan, the Chief technology Officer.

            Return the email as a JSON with a single key 'email_draft' and no preamble or explanation.

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    INITIAL_EMAIL: {initial_email} \n
    EMAIL_CATEGORY: {email_category} \n
    RESEARCH_INFO: {research_info} \n
    <|eot_id|><|start_header_id|>assistant<|end_header_id>""",
    input_variables=["initial_email", "email_category", "research_info"],
)

#create a chain for the above
chain_draft=draft_writer_prompt | llm | JsonOutputParser()
results_draft=chain_draft.invoke({"initial_email":EMAIL, "email_category": email_category, "research_info": research_result})

#print("\n Beggining Draft response below....\n")
#print("AI Draft:", results_draft)

rewrite_router_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert at evaluating emails that are draft emails for the customer and deciding if they
    need to be rewritten to be better. \n

    Use the following criteria to decide if the DRAFT_EMAIL needs to be rewritten: \n\n

    If the INITIAL_EMAIL only requires a simple response which the DRAFT_EMAIL contains, then it doesn't need to be rewritten.
    If the DRAFT_EMAIL addresses all the concerns of the INITIAL_EMAIL, then it doesn't need to be rewritten.
    If the DRAFT_EMAIL is missing information that the INITIAL_EMAIL requires, then it needs to be rewritten.

    Give a binary choice 'rewrite' (for needs to be rewritten) or 'no_rewrite' (for doesn't need to be rewritten) based on the DRAFT_EMAIL and the criteria.
    Return a JSON with a single key 'router_decision' and no preamble or explanation. \
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    INITIAL_EMAIL: {initial_email} \n
    EMAIL_CATEGORY: {email_category} \n
    DRAFT_EMAIL: {draft_email} \n
    <|eot_id|><|start_header_id|>assistant<|end_header_id>""",
    input_variables=["initial_email", "email_category", "draft_email"],
)

#chain of rewriting the draft email
rewrite_chain=rewrite_router_prompt | llm |JsonOutputParser()
output_rewrite=rewrite_chain.invoke({"initial_email":EMAIL, "email_category":email_category, "draft_email": results_draft})
#print("\n[+] Checking if rewrite is needed......\n")
#print(f"AI: {output_rewrite}")


#--Draft Email Analysis- Check whether there areas of improvement, and is the tone okay
draft_analysis_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are the Quality Control Agent. Read the INITIAL_EMAIL below from a human that has emailed \
    our company email address, the email_category that the categorizer agent gave it, and the \
    research from the research agent, and write an analysis of the email.

    Check if the DRAFT_EMAIL addresses the customer's issues based on the email category and the \
    content of the initial email.\n

    Give feedback on how the email can be improved and what specific things can be added or changed \
    to make the email more effective at addressing the customer's issues.

    You never make up or add information that hasn't been provided by the research_info or in the initial_email.

    Return the analysis as a JSON with a single key 'draft_analysis' and no preamble or explanation.

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    INITIAL_EMAIL: {initial_email} \n\n
    EMAIL_CATEGORY: {email_category} \n\n
    RESEARCH_INFO: {research_info} \n\n
    DRAFT_EMAIL: {draft_email} \n\n
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["initial_email", "email_category", "research_info", "draft_email"],
)

draft_analysis_chain = draft_analysis_prompt | llm | JsonOutputParser()

email_analysis=draft_analysis_chain.invoke({"initial_email": EMAIL, "email_category": email_category, "research_info": research_result, "draft_email":results_draft})
#print("\n[+]Draft email analysis....\n")
#print(f"AI: {email_analysis}")

#--Create a rewrite draft prompt to combine the changes by the analysis tool
rewrite_email_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are the Final Email Agent. Read the email analysis below from the QC Agent \
    and use it to rewrite and improve the draft_email to create a final email.

    You never make up or add information that hasn't been provided by the research_info or in the initial_email.

    Return the final email as JSON with a single key 'final_email' which is a string and no preamble or explanation.

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    EMAIL_CATEGORY: {email_category} \n\n
    RESEARCH_INFO: {research_info} \n\n
    DRAFT_EMAIL: {draft_email} \n\n
    DRAFT_EMAIL_FEEDBACK: {email_analysis} \n\n
    <|eot_id|>""",
    input_variables=["initial_email", "email_category", "research_info", "email_analysis", "draft_email"],
)

#create the chain
rewrite_email_chain= rewrite_email_prompt | llm | JsonOutputParser()
rewrite_email=rewrite_email_chain.invoke({"initial_email": EMAIL, "email_category": email_category, "research_info": research_result, "email_analysis":email_analysis,"draft_email":results_draft})

#print("\n[+]Rewritting email with changes....\n")
#print(f"AI: {rewrite_email}")

initial_email=EMAIL
draft_email_feedback=rewrite_email

#--------Lets Build Our Graph-------------------------#
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        initial_email: email
        email_category: email category
        draft_email: LLM generation
        final_email: LLM generation
        research_info: list of documents
        info_needed: whether to add search info
        num_steps: number of steps
    """
    initial_email:str
    email_category:str
    draft_email:str
    final_email:str
    research_info:List[str]
    info_needed:bool
    num_steps:int
    draft_email_feedback:dict


#categorize mail node of incoming email based content
def categorize_email(state):
    """Take the initial email and categorize it"""
    print("---Categorizing Initial EMail.....\n")
    initial_email=state['initial_email']
    num_steps=int(state['num_steps'])
    num_steps+=1

    #categorize the email
    email_category=email_category_generator.invoke({"initial_email":initial_email})
    print(email_category)
    #save the category to local disk
    write_markdown_file(email_category, "email_category")

    return {"email_category": email_category, "num_steps":num_steps}


#the research info search node-performs a web search based on keywords derived from the initial email 
def research_info_search(state):
    print("---Research Info Searching")
    initial_email=state["initial_email"]
    email_category=state["email_category"]
    research_info=state["research_info"]
    num_steps=state['num_steps']
    num_steps+=1

    #wev search for keywords
    keywords=search_keyword_chain.invoke({"initial_email":initial_email,
                                          "email_category":email_category})
    
    keywords=keywords['keywords']

    full_searches = []
    for keyword in keywords[:1]:  # Only taking the first keyword
        print(keyword)
        temp_docs = web_search_tool.invoke({"query": keyword})
        web_results = "\n".join([d["content"] for d in temp_docs])
        web_results = Document(page_content=web_results)
        if full_searches is not None:
            full_searches.append(web_results)
        else:
            full_searches = [web_results]
    print(full_searches)
    print(type(full_searches))
    
    return {"research_info": full_searches, "num_steps": num_steps}

def draft_email_writer(state):
    print("---WRITING DRAFT EMAIL---")
    # Implement logic to generate draft email based on research and category.
    return {"draft_email": draft_email_content, "num_steps": state['num_steps']}

def analyze_draft_email(state):
    print("---ANALYZING DRAFT EMAIL---")
    # Implement logic to analyze draft email, providing feedback.
    return {"draft_email_feedback": draft_feedback, "num_steps": state['num_steps']}

def rewrite_email(state):
    print("---REWRITING EMAIL---")
    # Use feedback to rewrite the email.
    return {"final_email": final_email_content, "num_steps": state['num_steps']}

def state_printer(state):
    print("---STATE---")
    print(state)
    return state
