#!/usr/bin/python3
import os
from typing import Type, ClassVar
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

# Load environment variables
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

# Pydantic models for tool arguments
class SimpleSearchInput(BaseModel):
    query: str = Field(description="should be a search query")

class MultiplyNumbersArgs(BaseModel):
    x: float = Field(description="First number to multiply")
    y: float = Field(description="Second number to multiply")

# Custom tool with only custom input
class SimpleSearchTool(BaseTool):
    name: ClassVar[str] = "simple_search"  # Fixed: Added type annotation
    description: ClassVar[str] = "useful when you want to answer questions about current events"  # Fixed: Added type annotation
    args_schema: Type[BaseModel] = SimpleSearchInput

    def _run(self, query: str) -> str:
        """Use the tool"""
        from tavily import TavilyClient
        
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "Error: TAVILY_API_KEY not found in environment variables"
            
        client = TavilyClient(api_key=api_key)
        try:
            results = client.search(query=query)
            return f"Search results for: {query}\n\n{results}\n"
        except Exception as e:
            return f"Error performing search: {str(e)}"

# Custom tool with custom input and output
class MultiplyNumbersTool(BaseTool):
    name: ClassVar[str] = "multiply_numbers"  # Fixed: Added type annotation
    description: ClassVar[str] = "useful for multiplying two numbers"  # Fixed: Added type annotation
    args_schema: Type[BaseModel] = MultiplyNumbersArgs

    def _run(self, x: float, y: float) -> str:
        """Use the tool"""
        try:
            result = x * y
            return f"The product of {x} and {y} is: {result}"
        except Exception as e:
            return f"Error calculating product: {str(e)}"

# Create tools using pydantic subclass approach
tools = [
    SimpleSearchTool(),
    MultiplyNumbersTool(),
]

# Initialize a chatOpenAI model
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com",
    temperature=0.3
)

# Pull the prompt template from the hub
prompt = hub.pull("hwchase17/openai-tools-agent")

# Create the agent using the create_tool_calling_agent function
agent = create_tool_calling_agent(
    llm=llm,    # model to use
    tools=tools, # list of tools available to the agent
    prompt=prompt # prompt template to guide the agent's response
)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,   # the agent to execute
    tools=tools,   # list of tools available to the agent
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3  # Added to prevent potential infinite loops
)

# Test the agent with sample queries
try:
    response = agent_executor.invoke({"input": "Search for Apple Intelligence"})
    print("Response for 'Search for Apple Intelligence':", response['output'])
except Exception as e:
    print(f"Error executing search: {str(e)}")

try:
    response = agent_executor.invoke({"input": "Multiply 10 and 20"})
    print("Response for 'Multiply 10 and 20':", response['output'])
except Exception as e:
    print(f"Error executing multiplication: {str(e)}")


"""


> Entering new AgentExecutor chain...

Invoking: `simple_search` with `{'query': 'Apple Intelligence'}`


Search results for: Apple Intelligence

{'query': 'Apple Intelligence', 'follow_up_questions': None, 'answer': None, 'images': [], 'results': [{'title': '10 Amazing Things You Can Do With Apple Intelligence On Your IPhone', 'url': 'https://www.forbes.com/sites/bernardmarr/2024/09/09/10-amazing-things-you-can-do-with-apple-intelligence-on-your-iphone/', 'content': "Apple's new AI system is set to transform how we use our iPhones. [+] unleashed creativity, Apple Intelligence brings a host of exciting features to your fingertips. Apple Intelligence is poised to revolutionize the iPhone experience, offering a suite of AI-powered tools that promise to make your digital life easier, more productive, and more creative. Apple Intelligence elevates your iPhone's ability to manage your time effectively by integrating information from various sources. Apple Intelligence can do that with a single tap, ensuring your message hits the right note every time. Simply describe what you want, and Apple Intelligence will generate it for you. By deeply integrating AI capabilities while maintaining a strong commitment to privacy, Apple is setting a new standard for personal computing.", 'score': 0.88705814, 'raw_content': None}, {'title': 'Apple Intelligence features expand to new languages and regions today', 'url': 'https://www.apple.com/newsroom/2025/03/apple-intelligence-features-expand-to-new-languages-and-regions-today/', 'content': 'Apple Intelligence, the personal intelligence system that delivers helpful and relevant intelligence while taking an extraordinary step forward for privacy in AI, is expanding to even more people around the world. Starting today, with the availability of iOS 18.4, iPadOS 18.4, and macOS Sequoia 15.4, Apple Intelligence features are now available in many new languages, including French, German', 'score': 0.8737439, 'raw_content': None}, {'title': 'How to get Apple Intelligence', 'url': 'https://support.apple.com/en-us/121115', 'content': 'Apple Apple Watch Apple Intelligence Explore All Apple Watch Why Apple Watch Shop Apple Watch Apple Watch Support Shop Apple Vision Pro Apple Vision Pro Support Apple TV Support Apple One Apple TV+ Support Apple Watch Made by Apple Apple Intelligence is available in beta starting with iOS 18.1, iPadOS 18.1, and macOS Sequoia 15.1 when your device language and Siri language are set 
to a supported language. Apple Intelligence requirements for iPhone, iPad, and Mac Apple Intelligence is available with macOS Sequoia 15.1 or later on supported Mac models. If traveling outside of the EU, Apple Intelligence will work on your iPhone or iPad if your language 
and Siri language are set to a supported language. Explore Apple Support Community', 'score': 0.769098, 'raw_content': None}, {'title': 'Apple Intelligence - Apple', 'url': 'https://www.apple.com/apple-intelligence/', 'content': 'Apple Apple Watch Apps by Apple Apps by Apple Apps by Apple Why Apple Watch Shop Apple Watch Apps by Apple Shop Apple Vision Pro Apple One Apple Watch Made by Apple * Apple\xa0Intelligence is available in beta on all iPhone\xa016 models, iPhone\xa015\xa0Pro, iPhone\xa015\xa0Pro\xa0Max, iPad\xa0mini (A17\xa0Pro), and 
iPad and Mac models with M1 and later, with Siri and device language set to English (Australia, Canada, Ireland, New Zealand, South Africa, UK, or U.S.), as part of an iOS\xa018, iPadOS\xa018, and macOS\xa0Sequoia software update. Apple One Or call 1-800-MY-APPLE.', 'score': 0.7606688, 'raw_content': None}, {'title': 'Apple Intelligence - Wikipedia', 'url': 'https://en.wikipedia.org/wiki/Apple_Intelligence', 'content': 'iOS / iPadOS Version[51]macOS Version[52]DateApple Intelligence Features18.1[53][54]15.1October 3, 2024Writing Tools (proofread, summarize, and rewrite only)Siri (new look, type to Siri, better with context only)Smart RepliesNotification SummaryPhotos Clean Up 
and Memory MakerReduce Interruptions Focus Mode18.2[55][56][57]15.2December 11, 2024More Writing Tools (compose, describe changes with ChatGPT)Image PlaygroundImage WandSiri (ChatGPT integration only)Mail Categorization[58] (iPhone)Genmoji (iPhone and iPad only for now)Visual Intelligence (iPhone 16 and 16 Pro or later)18.4 Beta 2[59][60]15.4 Beta 2March 3, 2025Priority NotificationsMail Categorization[61] (iPad and Mac)Image Playground SketchVisual Intelligence[35] (iPhone 15 Pro/Pro Max and iPhone 16e)', 'score': 0.7382357, 'raw_content': None}], 'response_time': 1.89}
Here are some resources about Apple Intelligence:

1. [10 Amazing Things You Can Do With Apple Intelligence On Your iPhone](https://www.forbes.com/sites/bernardmarr/2024/09/09/10-amazing-things-you-can-do-with-apple-intelligence-on-your-iphone/) - This article discusses how Apple's AI system transforms iPhone usage with features that enhance productivity and creativity.

2. [Apple Intelligence features expand to new languages and regions today](https://www.apple.com/newsroom/2025/03/apple-intelligence-features-expand-to-new-languages-and-regions-today/) - Apple Intelligence is expanding globally with support for more languages and regions, 
available with updates to iOS, iPadOS, and macOS.

3. [How to get Apple Intelligence](https://support.apple.com/en-us/121115) - A guide on accessing Apple Intelligence, including device and software requirements.

4. [Apple Intelligence - Apple](https://www.apple.com/apple-intelligence/) - Official Apple page detailing the availability and features 
of Apple Intelligence on supported devices.

5. [Apple Intelligence - Wikipedia](https://en.wikipedia.org/wiki/Apple_Intelligence) - A Wikipedia page summarizing the features and updates of Apple Intelligence across various software versions.

Let me know if you'd like more details on any of these!

> Finished chain.
Response for 'Search for Apple Intelligence': Here are some resources about Apple Intelligence:

1. [10 Amazing Things You Can Do With Apple Intelligence On Your iPhone](https://www.forbes.com/sites/bernardmarr/2024/09/09/10-amazing-things-you-can-do-with-apple-intelligence-on-your-iphone/) - This article discusses how Apple's AI system transforms iPhone usage with features that enhance productivity and creativity.

2. [Apple Intelligence features expand to new languages and regions today](https://www.apple.com/newsroom/2025/03/apple-intelligence-features-expand-to-new-languages-and-regions-today/) - Apple Intelligence is expanding globally with support for more languages and regions, 
available with updates to iOS, iPadOS, and macOS.

3. [How to get Apple Intelligence](https://support.apple.com/en-us/121115) - A guide on accessing Apple Intelligence, including device and software requirements.

4. [Apple Intelligence - Apple](https://www.apple.com/apple-intelligence/) - Official Apple page detailing the availability and features 
of Apple Intelligence on supported devices.

5. [Apple Intelligence - Wikipedia](https://en.wikipedia.org/wiki/Apple_Intelligence) - A Wikipedia page summarizing the features and updates of Apple Intelligence across various software versions.

Let me know if you'd like more details on any of these!


> Entering new AgentExecutor chain...

Invoking: `multiply_numbers` with `{'x': 10, 'y': 20}`


The product of 10.0 and 20.0 is: 200.0The product of 10 and 20 is 200.

> Finished chain.
Response for 'Multiply 10 and 20': The product of 10 and 20 is 200.
"""