from langgraph.graph import StateGraph, END
import time

from . import GenerateAgent, SearchAgent, CurateAgent, WriteAgent, Config, ResearchState, InputState, OutputState


class MasterAgent:
    def __init__(self):
        cfg = Config()

        # Initialize agents
        generate_agent = GenerateAgent(cfg)
        search_agent = SearchAgent(cfg)
        curate_agent = CurateAgent(cfg)
        write_agent = WriteAgent(cfg)

        # Define a Langchain graph
        workflow = StateGraph(ResearchState, input=InputState, output=OutputState)

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

    async def run(self, query: str):
        start_time = time.time()
        print(f"Starting research process for query: {query}")

        # compile the graph
        graph = self.workflow.compile()

        # invoke the graph
        await graph.ainvoke({"query": query})

        end_time = time.time()
        duration = end_time - start_time
        print(f"Research process completed in {duration:.2f} seconds")

    def compile(self):
        # compile the graph and return it (for LangGraph Studio)
        graph = self.workflow.compile()
        return graph

