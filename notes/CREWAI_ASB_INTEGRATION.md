# CrewAI inside official ASB: what is required

This note explains the architectural issue.

## Why the official ASB scripts do not automatically run CrewAI

ASB's public quickstart is config-driven and launches:
- `scripts/agent_attack.py`
- then `main_attacker.py`
- then ASB/AIOS agent internals

That means the official ASB pipeline expects the attacked target to be an agent implemented in ASB's own stack.

## What you must change for a true official integration

You would need to do one of these:

### Path 1: Wrap CrewAI as an ASB-compatible agent
Create a new ASB agent class that:
- accepts the task input from ASB
- executes your CrewAI two-agent workflow
- returns outputs in the format ASB expects
- exposes tool execution / memory behavior in a way ASB can attack and log

### Path 2: Replace ASB's agent execution layer
Modify the relevant ASB runtime so that instead of using the current agent implementation, it calls your CrewAI workflow.

This is a larger code change and is usually harder to maintain.

## Practical recommendation

For most research projects, use:
- official ASB scripts for baseline reference runs
- a separate CrewAI replay harness for your actual CrewAI system

Then report this clearly in your methodology.
