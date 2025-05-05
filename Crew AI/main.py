#!/usr/bin/python3
from dotenv import load_dotenv
from crewai import Crew
from tasks import MeetingPrepTasks
from agents import MeetingPrepAgents
from langchain_openai import ChatOpenAI  # New import
import os

def main():
    load_dotenv()

    print("******************"*6)
    print("           Meeting Prep Crew    ")
    print("******************"*6)
    
    # Initialize ChatGPT
    load_dotenv()
    api_key = os.getenv("GITHUB_TOKEN")

    # Create the LangChain chat model using the GitHub Marketplace endpoint
    llm = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=api_key,
        base_url="https://models.inference.ai.azure.com"
    )
    
    meeting_participants = input("What are the emails for the participants(other than you) in the meeting?\n")
    meeting_context = input("What is the context of the meeting?\n")
    meeting_objective = input("What is your objective for this meeting?\n")

    tasks = MeetingPrepTasks()
    agents = MeetingPrepAgents(llm)  # Pass the LLM to your agents

    # Create the agents
    research_agent = agents.research_agent()
    industry_analysis_agent = agents.industry_analysis_agent()
    meeting_strategy_agent = agents.meeting_strategy_agent()
    summary_and_briefing_agent = agents.summary_and_briefing_agent()

    # Create our tasks
    research_task = tasks.research_task(research_agent, meeting_participants, meeting_context)
    industry_analysis_task = tasks.industry_analysis_task(industry_analysis_agent, meeting_participants, meeting_context)
    meeting_strategy_task = tasks.meeting_strategy_task(meeting_strategy_agent, meeting_context, meeting_objective)
    summary_and_briefing_task = tasks.summary_and_briefing_task(summary_and_briefing_agent, meeting_context, meeting_objective)
     
    meeting_strategy_task.context = [research_task, industry_analysis_task]
    summary_and_briefing_task.context = [research_task, industry_analysis_task, meeting_strategy_task]

    # Create the crew
    crew = Crew(
        agents=[
            research_agent,
            industry_analysis_agent,
            meeting_strategy_agent,
            summary_and_briefing_agent
        ],
        tasks=[
            research_task,
            industry_analysis_task,
            meeting_strategy_task,
            summary_and_briefing_task
        ]
    )

    # Get results
    results = crew.kickoff()
    print("\n\n########################")
    print("## Here is your Meeting Prep")
    print("########################\n")
    print(results)

if __name__ == "__main__":
    main()