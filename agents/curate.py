from tavily import AsyncTavilyClient
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from typing import List
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio

from .memory.research import ResearchState

class RankedSource(BaseModel):
    url: str = Field(description="The URL of the source")
    rank: int = Field(description="Rank of the source (1 being the highest)")

class TavilyExtractInput(BaseModel):
    ranked_sources: List[RankedSource] = Field(description="List of ranked sources, "
                                                           "ordered by relevance, trustworthiness, and reliability")

class CurateAgent:
    def __init__(self, cfg):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.tavily_client = AsyncTavilyClient()
        self.cfg = cfg

    async def run(self, state: ResearchState):
        print("In curate agent")
        state = state.model_dump()
        research_depth = state.get('research_depth')
        if research_depth == "advanced":
            # Original ranking logic for detailed research
            system_prompt = f"""Today's date is {datetime.now().strftime('%d/%m/%Y')}.\n
            {state['agent']['prompt']}.\n
            Your current task is to review a list of documents and select the most relevant, trusted, and reliable sources 
            related to the following research task: {state['query']}.\n
            
            Please follow these guidelines:
            1. Evaluate each source based on its relevance to the query, credibility, and reliability.
            2. Consider factors such as the author's expertise, the publication's reputation, 
                and the recency of the information.
            3. Rank the sources in order of their overall quality and relevance, with 1 being the highest rank.
            4. Provide a brief reason for each ranking.
            5. Select up to {self.cfg.MAX_CURATED_DOCS} of the best sources.

            Here is the list of documents gathered for your review:\n{state['research_data']}\n\n
            
            Respond with a ranked list of the best sources, including their URLs and ranks.
            """
            messages = [SystemMessage(content=system_prompt)]
            ranked_sources = self.model.with_structured_output(TavilyExtractInput).invoke(messages)
            print(f"Selected and ranked the following sources:")
            for source in ranked_sources.ranked_sources:
                print(f"Rank {source.rank}: {source.url}")

            # Create a dictionary of relevant documents based on the ranked sources
            curated_data = {source.url: state['research_data'][source.url] for source in
                            ranked_sources.ranked_sources if source.url in state['research_data']}
            urls = [source.url for source in ranked_sources.ranked_sources]
        else:
            # If research_depth is basic, skip ranking and use all sources (without extracting sources)
            print("Basic research mode - skipping ranking")
            curated_data = state['research_data']
            # urls = list(curated_data.keys())
            urls = []

        msg = ""
        # Rest of the code remains the same, but use urls list instead of ranked_sources
        async def process_batch(url_batch):
            batch_msg = ""
            try:
                response = await self.tavily_client.extract(urls=url_batch)
                for itm in response['results']:
                    url = itm['url']
                    raw_content = itm['raw_content']
                    curated_data[url]['raw_content'] = raw_content
                    batch_msg += f"{url}\n"
                return batch_msg
            except Exception as e:
                return f"Error occurred during Tavily Extract request for batch: {e}\n"

        # Split URLs into batches of 20
        url_batches = [urls[i:i + 20] for i in range(0, len(urls), 20)]

        # Process all batches in parallel
        results = await asyncio.gather(*[process_batch(batch) for batch in url_batches])

        # Collect messages from all batches
        msg += "Extracted raw content for:\n" + "".join(results)
        print(msg)
        return {"curated_data": curated_data}
