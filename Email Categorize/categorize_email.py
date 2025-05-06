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
result=email_category_generator.invoke({"initial_email":EMAIL})
print(f"AI: {result}")
