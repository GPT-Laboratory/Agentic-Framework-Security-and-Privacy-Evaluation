"""
This file shows the exact part you replace with your real two CrewAI agents.
Copy the agent definitions into src/crew_app.py if needed.
"""

from crewai import Agent


def build_your_agents(model: str):
    agent_1 = Agent(
        role="Your Agent 1",
        goal="Do the first stage of the business workflow safely.",
        backstory="Your real first CrewAI agent definition goes here.",
        llm=model,
        verbose=True,
        allow_delegation=False,
    )

    agent_2 = Agent(
        role="Your Agent 2",
        goal="Review the first agent output and safely complete the workflow.",
        backstory="Your real second CrewAI agent definition goes here.",
        llm=model,
        verbose=True,
        allow_delegation=False,
    )
    return agent_1, agent_2
