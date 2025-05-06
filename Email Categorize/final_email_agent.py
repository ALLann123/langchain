#!/usr/bin/python3
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from pprint import pprint
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import List, Dict, Any

load_dotenv()
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)

# Define the state structure
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        initial_email: The original email content
        email_category: The categorized type of email
        draft_email: First draft of the response
        final_email: Final version of the response
        research_info: Information gathered from research
        info_needed: Whether additional info is needed
        num_steps: Counter of steps taken
        draft_email_feedback: Analysis of the draft email
        router_decision: Decision from routing steps
    """
    initial_email: str
    email_category: str
    draft_email: Dict[str, str]
    final_email: Dict[str, str]
    research_info: List[Document]
    info_needed: bool
    num_steps: int
    draft_email_feedback: Dict[str, str]
    router_decision: str

# Helper functions
def write_markdown_file(content, filename):
    """Writes content to a markdown file."""
    with open(f"{filename}.md", "w") as f:
        f.write(content)

# Define all the prompts
categorize_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an Email Categorizer Agent. You are a master at understanding what a customer wants when they write an email and are able to categorize it in a useful way.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Conduct a comprehensive analysis of the email provided and categorize it into one of the following categories:
        price_enquiry - used when someone is asking for information about pricing
        customer_complaint - used when someone is complaining about something
        product_enquiry - used when someone is asking for information about a product feature, benefit, or service but not about pricing
        customer_feedback - used when someone is giving feedback about a product
        off_topic - when it doesn't relate to any other category.
    Output a single category only from the types ('price_enquiry', 'customer_complaint', 'product_enquiry', 'customer_feedback', 'off_topic')
    EMAIL CONTENT:\n\n {initial_email} \n\n
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["initial_email"],
)

research_router_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert at reading the initial email and routing web search
    or directly to a draft email.
    Use the following criteria to decide how to route the email:
    If the initial email only requires a simple response:
    - Choose 'draft_email' for questions you can easily answer
    - If the email is just saying thank you, etc., choose 'draft_email'
    Otherwise, use 'research_info'.
    Given a binary choice 'research_info' or 'draft_email' based on the question.
    Return a JSON with a single key 'router_decision' and no preamble or explanation.
    Use both the initial email and email category to make your decision.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Email to route INITIAL_EMAIL: {initial_email}
    EMAIL_CATEGORY: {email_category}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["initial_email", "email_category"]
)

search_keyword_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a master at working out the best keywords for a web search
    to find the most relevant information for the customer.
    Given the INITIAL_EMAIL and EMAIL_CATEGORY, work out the best 
    keywords that will find the most relevant information to help write
    the final email.
    Return a JSON with a single key 'keywords' containing no more than
    3 keywords, and no preamble or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    INITIAL_EMAIL:{initial_email}
    EMAIL_CATEGORY:{email_category}
    <|eot_id|><|start_header_id|>assistant<|end_header_id>""",
    input_variables=["initial_email", "email_category"],     
)

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

rewrite_router_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert at evaluating emails that are draft emails for the customer and deciding if they
    need to be rewritten to be better.
    Use the following criteria to decide if the DRAFT_EMAIL needs to be rewritten:
    If the INITIAL_EMAIL only requires a simple response which the DRAFT_EMAIL contains, then it doesn't need to be rewritten.
    If the DRAFT_EMAIL addresses all the concerns of the INITIAL_EMAIL, then it doesn't need to be rewritten.
    If the DRAFT_EMAIL is missing information that the INITIAL_EMAIL requires, then it needs to be rewritten.
    Give a binary choice 'rewrite' (for needs to be rewritten) or 'no_rewrite' (for doesn't need to be rewritten) based on the DRAFT_EMAIL and the criteria.
    Return a JSON with a single key 'router_decision' and no preamble or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    INITIAL_EMAIL: {initial_email} \n
    EMAIL_CATEGORY: {email_category} \n
    DRAFT_EMAIL: {draft_email} \n
    <|eot_id|><|start_header_id|>assistant<|end_header_id>""",
    input_variables=["initial_email", "email_category", "draft_email"],
)

draft_analysis_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are the Quality Control Agent. Read the INITIAL_EMAIL below from a human that has emailed \
    our company email address, the email_category that the categorizer agent gave it, and the \
    research from the research agent, and write an analysis of the email.
    Check if the DRAFT_EMAIL addresses the customer's issues based on the email category and the \
    content of the initial email.
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
    input_variables=["email_category", "research_info", "email_analysis", "draft_email"],
)

# Create all the chains
email_category_generator = categorize_prompt | llm | StrOutputParser()
research_router = research_router_prompt | llm | JsonOutputParser()
search_keyword_chain = search_keyword_prompt | llm | JsonOutputParser()
draft_writer_chain = draft_writer_prompt | llm | JsonOutputParser()
rewrite_router_chain = rewrite_router_prompt | llm | JsonOutputParser()
draft_analysis_chain = draft_analysis_prompt | llm | JsonOutputParser()
rewrite_email_chain = rewrite_email_prompt | llm | JsonOutputParser()

# Define nodes for the graph
def categorize_email(state: GraphState) -> Dict[str, Any]:
    """Categorize the incoming email."""
    print("---CATEGORIZING EMAIL---")
    initial_email = state["initial_email"]
    email_category = email_category_generator.invoke({"initial_email": initial_email})
    write_markdown_file(email_category, "email_category")
    return {"email_category": email_category}

def route_to_research_or_draft(state: GraphState) -> Dict[str, Any]:
    """Decide whether to research or draft directly."""
    print("---ROUTING TO RESEARCH OR DRAFT---")
    initial_email = state["initial_email"]
    email_category = state["email_category"]
    router_decision = research_router.invoke({
        "initial_email": initial_email,
        "email_category": email_category
    })
    return {"router_decision": router_decision["router_decision"]}

def research_info(state: GraphState) -> Dict[str, Any]:
    """Perform research based on email content."""
    print("---RESEARCHING INFORMATION---")
    initial_email = state["initial_email"]
    email_category = state["email_category"]
    
    # Get keywords
    keywords = search_keyword_chain.invoke({
        "initial_email": initial_email,
        "email_category": email_category
    })["keywords"]
    
    # Simulate web search (in a real implementation, you'd use a web search tool)
    research_results = [
        Document(page_content=f"Research result for keyword: {keyword}") 
        for keyword in keywords[:1]  # Just use first keyword for demo
    ]
    
    return {"research_info": research_results}

def draft_email(state: GraphState) -> Dict[str, Any]:
    """Draft the initial email response."""
    print("---DRAFTING EMAIL---")
    initial_email = state["initial_email"]
    email_category = state["email_category"]
    research_info = state.get("research_info", [])
    
    # Convert research docs to string for the prompt
    research_content = "\n".join([doc.page_content for doc in research_info])
    
    draft = draft_writer_chain.invoke({
        "initial_email": initial_email,
        "email_category": email_category,
        "research_info": research_content
    })
    return {"draft_email": draft}

def route_to_rewrite_or_final(state: GraphState) -> Dict[str, Any]:
    """Decide whether the draft needs rewriting."""
    print("---CHECKING IF REWRITE NEEDED---")
    initial_email = state["initial_email"]
    email_category = state["email_category"]
    draft_email = state["draft_email"]
    
    decision = rewrite_router_chain.invoke({
        "initial_email": initial_email,
        "email_category": email_category,
        "draft_email": draft_email
    })
    return {"router_decision": decision["router_decision"]}

def analyze_draft(state: GraphState) -> Dict[str, Any]:
    """Analyze the draft email for improvements."""
    print("---ANALYZING DRAFT---")
    initial_email = state["initial_email"]
    email_category = state["email_category"]
    research_info = state.get("research_info", [])
    draft_email = state["draft_email"]
    
    # Convert research docs to string for the prompt
    research_content = "\n".join([doc.page_content for doc in research_info])
    
    analysis = draft_analysis_chain.invoke({
        "initial_email": initial_email,
        "email_category": email_category,
        "research_info": research_content,
        "draft_email": draft_email
    })
    return {"draft_email_feedback": analysis}

def rewrite_email(state: GraphState) -> Dict[str, Any]:
    """Rewrite the email based on feedback."""
    print("---REWRITING EMAIL---")
    email_category = state["email_category"]
    research_info = state.get("research_info", [])
    draft_email = state["draft_email"]
    draft_email_feedback = state["draft_email_feedback"]
    
    # Convert research docs to string for the prompt
    research_content = "\n".join([doc.page_content for doc in research_info])
    
    final_email = rewrite_email_chain.invoke({
        "email_category": email_category,
        "research_info": research_content,
        "draft_email": draft_email,
        "email_analysis": draft_email_feedback["draft_analysis"]
    })
    return {"final_email": final_email}

def final_response(state: GraphState) -> Dict[str, Any]:
    """Return the final response."""
    print("---FINAL RESPONSE---")
    return {"final_email": state["draft_email"]}

# Build the workflow graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("categorize_email", categorize_email)
workflow.add_node("route_to_research_or_draft", route_to_research_or_draft)
workflow.add_node("research_info", research_info)
workflow.add_node("draft_email", draft_email)
workflow.add_node("route_to_rewrite_or_final", route_to_rewrite_or_final)
workflow.add_node("analyze_draft", analyze_draft)
workflow.add_node("rewrite_email", rewrite_email)
workflow.add_node("final_response", final_response)

# Add edges
workflow.add_edge("categorize_email", "route_to_research_or_draft")
workflow.add_conditional_edges(
    "route_to_research_or_draft",
    lambda x: x["router_decision"],
    {
        "research_info": "research_info",
        "draft_email": "draft_email"
    }
)
workflow.add_edge("research_info", "draft_email")
workflow.add_edge("draft_email", "route_to_rewrite_or_final")
workflow.add_conditional_edges(
    "route_to_rewrite_or_final",
    lambda x: x["router_decision"],
    {
        "rewrite": "analyze_draft",
        "no_rewrite": "final_response"
    }
)
workflow.add_edge("analyze_draft", "rewrite_email")
workflow.add_edge("rewrite_email", "final_response")

# Set entry and end points
workflow.set_entry_point("categorize_email")
workflow.set_finish_point("final_response")

# Compile the graph
email_agent = workflow.compile()

# Example usage
if __name__ == "__main__":
    test_email = """HI there, 
    I am emailing to say that I had a wonderful stay at your resort last week.
    I really appreciate what your staff did.

    Thanks, 
    Paul
    """
    
    # Initialize the state
    initial_state = {
        "initial_email": test_email,
        "email_category": "",
        "draft_email": {},
        "final_email": {},
        "research_info": [],
        "info_needed": False,
        "num_steps": 0,
        "draft_email_feedback": {},
        "router_decision": ""
    }
    
    # Run the agent
    result = email_agent.invoke(initial_state)
    print("\nFinal Email Response:")
    print(result["final_email"]["final_email"])