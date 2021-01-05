from .agent1 import Agent1


def get_agent(name):
    if name == "agent1":
        agent = Agent1()
    else:
        raise ValueError(f"Unknown Agent: {name}")

    return agent
