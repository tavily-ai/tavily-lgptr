from tavily import AsyncTavilyClient
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from typing import List
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio

from memory.research import ResearchState

class TavilyExtractInput(BaseModel):
    urls: List[str] = Field(description="list of a single or several URLs for extracting raw content to gather additional information")

class CurateAgent:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.tavily_client = AsyncTavilyClient()

    async def run(self, state: ResearchState):
        # TODO maybe change this to use a better rank & filter mechanism
        """
        Selects the most relevant search documents for curation, then uses Tavily Extract to obtain additional context.
        :param state:
        :return:
        """
        print("In curate agent")
        system_prompt = f"""Today's date is {datetime.now().strftime('%d/%m/%Y')}.\n
        {state['agent']['prompt']}.\n
        Your current task is to review a list of documents and select the most relevant URLs related to the following research task: {state['task']['query']}.\n
        Here is the list of documents gathered for your review:\n{state['research_data']}\n\n"""
        messages = [SystemMessage(content=system_prompt)]
        relevant_urls = self.model.with_structured_output(TavilyExtractInput).invoke(messages)
        print(f"Selected the following urls {relevant_urls.urls}")

        # Create a dictionary of relevant documents based on the URLs returned by the model
        curated_data = {url: state['research_data'][url] for url in relevant_urls.urls if url in state['research_data']}
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
        url_batches = [relevant_urls.urls[i:i + 20] for i in range(0, len(relevant_urls.urls), 20)]

        # Process all batches in parallel
        results = await asyncio.gather(*[process_batch(batch) for batch in url_batches])

        # Collect messages from all batches
        msg += "Extracted raw content for:\n" + "".join(results)
        print(msg)
        return {"curated_data": curated_data}


