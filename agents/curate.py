from tavily import AsyncTavilyClient
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from typing import List
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio

from memory.research import ResearchState

class RankedSource(BaseModel):
    url: str = Field(description="The URL of the source")
    rank: int = Field(description="Rank of the source (1 being the highest)")

class TavilyExtractInput(BaseModel):
    ranked_sources: List[RankedSource] = Field(description="List of ranked sources, ordered by relevance, trustworthiness, and reliability")

class CurateAgent:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.tavily_client = AsyncTavilyClient()

    async def run(self, state: ResearchState):
        print("In curate agent")
        system_prompt = f"""Today's date is {datetime.now().strftime('%d/%m/%Y')}.\n
        {state['agent']['prompt']}.\n
        Your current task is to review a list of documents and select the most relevant, trusted, and reliable sources related to the following research task: {state['task']['query']}.\n
        
        Please follow these guidelines:
        1. Evaluate each source based on its relevance to the query, credibility, and reliability.
        2. Consider factors such as the author's expertise, the publication's reputation, the recency of the information, and the presence of citations or references.
        3. Rank the sources in order of their overall quality and relevance, with 1 being the highest rank.
        4. Provide a brief reason for each ranking.
        5. Select up to 10 of the best sources.

        Here is the list of documents gathered for your review:\n{state['research_data']}\n\n
        
        Respond with a ranked list of the best sources, including their URLs and ranks.
        """
        messages = [SystemMessage(content=system_prompt)]
        ranked_sources = self.model.with_structured_output(TavilyExtractInput).invoke(messages)
        print(f"Selected and ranked the following sources:")
        for source in ranked_sources.ranked_sources:
            print(f"Rank {source.rank}: {source.url}")

        # Create a dictionary of relevant documents based on the ranked sources
        curated_data = {source.url: state['research_data'][source.url] for source in ranked_sources.ranked_sources if source.url in state['research_data']}
        msg = ""

        # Process URLs in batches of 20
        async def process_batch(url_batch):
            batch_msg = ""
            try:
                # Extract raw content from the selected URLs using the Tavily client
                response = await self.tavily_client.extract(urls=url_batch)

                # Save the raw content into the RAG_docs dictionary for each URL
                for itm in response['results']:
                    url = itm['url']
                    raw_content = itm['raw_content']
                    curated_data[url]['raw_content'] = raw_content
                    batch_msg += f"{url}\n"
                return batch_msg
            except Exception as e:
                # Handle errors for this batch
                return f"Error occurred during Tavily Extract request for batch: {e}\n"

        # Split URLs into batches of 20
        url_batches = [source.url for source in ranked_sources.ranked_sources]
        url_batches = [url_batches[i:i + 20] for i in range(0, len(url_batches), 20)]

        # Process all batches in parallel
        results = await asyncio.gather(*[process_batch(batch) for batch in url_batches])

        # Collect messages from all batches
        msg += "Extracted raw content for:\n" + "".join(results)
        print(msg)
        return {"curated_data": curated_data}
