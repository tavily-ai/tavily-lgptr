from langgraph.graph import StateGraph, END

from . import GenerateAgent, SearchAgent, CurateAgent, WriteAgent, Config, ResearchState


class MasterAgent:
    def __init__(self):
        cfg = Config()
        # Initialize agents
        generate_agent = GenerateAgent()
        search_agent = SearchAgent(cfg.MAX_SEARCH_QUERIES)
        curate_agent = CurateAgent(cfg.MAX_CURATED_DOCS)
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
        graph = self.workflow.compile()

        # just invoke
        await graph.ainvoke({"task": task})

    def compile(self):
        # compile the graph and return it (for LangGraph Studio)
        graph = self.workflow.compile()
        return graph

