import asyncio
from langgraph.graph import StateGraph, END, add_messages

from generate import GenerateAgent
from search import SearchAgent
from curate import CurateAgent
from write import WriteAgent
from memory.research import ResearchState

from dotenv import load_dotenv
load_dotenv('../.env.local')
class MasterAgent:
    def __init__(self):
        # Initialize agents
        generate_agent = GenerateAgent()
        search_agent = SearchAgent()
        curate_agent = CurateAgent()
        write_agent = WriteAgent()

        # Define a Langchain graph
        workflow = StateGraph(ResearchState)

        # Add nodes for each agent
        workflow.add_node('generate', generate_agent.run)
        workflow.add_node('search', search_agent.run)
        workflow.add_node('curate', curate_agent.run)
        workflow.add_node('write', write_agent.run)

        # Set up edges
        workflow.add_edge('generate', 'search')
        workflow.add_edge('search', 'curate')
        workflow.add_edge('curate', 'write')
        workflow.add_edge('write', END)

        # set up start and end nodes
        workflow.set_entry_point('generate')

        self.workflow = workflow

    async def run(self, task: dict):

        # compile the graph
        chain = self.workflow.compile()

        # stram events
        # async for event in chain.astream_events({"task": task}, version="v2"):
        #     if event["event"] == 'on_chain_start':
        #         print(event)

        # just invoke
        await chain.ainvoke({"task": task})

async def main():
    master_agent = MasterAgent()
    _query = input("What do you want to research?\n")
    task = {
        "query": _query
    }
    await master_agent.run(task)

if __name__ == "__main__":
    asyncio.run(main())