#!/usr/bin/python3
from langchain_groq import ChatGroq # type: ignore
from dotenv import load_dotenv # type: ignore
import os
from pprint import pprint
from langchain.prompts import PromptTemplate # type: ignore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)


#test the chain above
EMAIL="""HI there, \n
I am emailing to say that I had a wonderful stay at your resort last week.\n
I really appreciate what yor staff did

Thanks, 
Paul
"""

email_category="customer feedback"
results_draft="Yoh, we cannt get back to you!!"

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
print("\n[+] Checking if rewrite is needed......\n")
print(f"AI: {output_rewrite}")
