from tavily import AsyncTavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timezone
import asyncio

from .memory.research import ResearchState


# Define Tavily's arguments to enhance the search dynamism
class TavilyQuery(BaseModel):
    query: str = Field(description="web search query")
    topic: str = Field(description="type of search, should be 'general' or 'news'")
    days: int = Field(description="number of days back to run 'news' search", default=3)
    domains: Optional[List[str]] = Field(default=None,
                                         description="list of domains to include in the research. Useful when trying to gather information from trusted and relevant domains")


# Define Input for Tavily Search using a multi-query strategy to enhance query precision and enable more focused results from Tavily
class TavilySearchInput(BaseModel):
    sub_queries: List[TavilyQuery] = Field(description="set of sub-queries that can be answered in isolation")


class SearchAgent:
    def __init__(self, max_queries):
        self.model = ChatOpenAI(model="gpt-4o", temperature=0.4)
        self.tavily_client = AsyncTavilyClient()
        self.MAX_QUERIES = max_queries

    async def tavily_search(self, sub_queries: List[TavilyQuery]):
        """Perform searches for each sub-query using the Tavily search tool concurrently."""

        # Define a coroutine function to perform a single search with error handling
        async def perform_search(itm):
            try:
                # Add date to the query as we need the most recent results
                query_with_date = f"{itm.query} {datetime.now().strftime('%m-%Y')}"
                # Attempt to perform the search
                response = await self.tavily_client.search(query=query_with_date, topic=itm.topic, days=itm.days,
                                                           max_results=10)
                return response['results']
            except Exception as e:
                # Handle any exceptions, log them, and return an empty list
                print(f"Error occurred during search for query '{itm.query}': {str(e)}")
                return []

            # Run all the search tasks in parallel

        search_tasks = [perform_search(itm) for itm in sub_queries]
        search_responses = await asyncio.gather(*search_tasks)

        # Combine the results from all the responses
        search_results = []
        for response in search_responses:
            search_results.extend(response)

        return search_results

    async def generate_search_queries(self, agent, query):
        """
        Generate search queries using the agent's persona and the initial query.

        :param agent: A dictionary with the agent's name and prompt.
        :param query: The original user query.
        :return: A list of search queries (structured in a list of TavilyQuery instances).
        """

        initial_search_results = await self.tavily_search([TavilyQuery(query=query, topic='general')])

        system_prompt = f"""
            {agent['prompt']}
            You are a tasked tasked with generating {self.MAX_QUERIES-1} search queries to find relevant information for the following task: "{query}".
            Context: {initial_search_results}
            
            Use this context to inform and refine your search queries. 
            The context provides real-time web information that can help you generate more specific and relevant queries. 
            Consider any current events, recent developments, or specific details mentioned in the context that could enhance the search queries.
            
            Assume the current date is {datetime.now(timezone.utc).strftime('%B %d, %Y')} if required.
            The response should contain ONLY the list of search queries.
        """
        messages = [SystemMessage(content=system_prompt)]
        try:
            response = await self.model.with_structured_output(TavilySearchInput).ainvoke(messages)
            return response.sub_queries, initial_search_results
        except Exception as e:
            print("⚠️ Error in generating search queries to Tavily")
            return [], initial_search_results

    async def run(self, state: ResearchState):
        print("In search agent")
        sub_queries, initial_search_results = await self.generate_search_queries(state['agent'], state['task']['query'])
        search_results = await self.tavily_search(sub_queries)
        search_results.extend(initial_search_results)
        # Save search results
        docs = state.get('research_data', {})
        for doc in search_results:
            # Make sure that this document was not retrieved before
            if doc['url'] not in search_results:
                docs[doc['url']] = doc
        return {"research_data": docs}
