#!/usr/bin/python3
from textwrap import dedent
from crewai import Task # type: ignore

class MeetingPrepTasks():
    def research_task(self,agent, meeting_participants, meeting_context):
        return Task(
            description=dedent(f"""\
                Conduct comprehensive research on each of the individuals and companies
                involved in the upcoming meeting. Gather information on recent
                news, achievements, professional background, and any relevant
                business activities.
                               
                Participants:{meeting_participants}
                Meeting Context:{meeting_context}"""),
            expected_output=dedent("""\
                A detailed report summarizing key findings about each participant
                and company, highlighting information that could be relevant for the meeting."""),
            async_execution=True,
            agent=agent
        )

    def industry_analysis_task(self, agent, meeting_participants, meeting_context):
        return Task(
            description=dedent(f"""\
                Analyze current industry trends, challenges, and opportunities 
                relevant to the meeting's context. Consider market reports, recent
                developers, and expert opinions to provide a comprehensive
                overview of the industry landscape.text
                Participants:{meeting_participants}
                Meeting context: {meeting_context}"""),

            expected_output=dedent("""\
                An insightful analysis that identifies major trends, potential challenges and
                strategic opportunities."""),
                async_execution=True,
                agent=agent
        )
    
    def meeting_strategy_task(self, agent, meeting_context, meeting_objective):
        return Task(
            description=dedent(f"""\
                Develop strategic talking points, questions, and discusiopn anlges 
                for the meeting based on the research and industry analysis conducted
                Meeting Context: {meeting_context}
                Meeting objective:{meeting_objective}"""),
            expectect_output=dedent(f"""\
                COmplete report with a list of key talking points, strategic  questions
                to ask to help achieve the meetings objective during the meeting.""" ),
            agent=agent
        )

    def summary_and_briefing_task(self, agent, meeting_context, meeting_objective):
        return Task(
            description=dedent(f"""\
            Complete all the research findings, industry analysis, and strategic
            talking points into concise, comprehensice briefing document for 
            the meeting
            Ensure the briefing is easy to digest and equiips the meeting participants with all necessary information and strategies.
            Meeting ontext:{meeting_context}
            Meeting Objective:{meeting_objective}"""),
            expected_output=dedent("""\
            A well-structured briefing doocument that includes sections for
            participants bios, industry overview, talking points, and 
            strategic reccomendations."""),
            agent=agent
        )


