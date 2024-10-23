from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
from memory.research import ResearchState

class Citation(BaseModel):
    source_id: str = Field(description="The url of a SPECIFIC source which justifies the answer.")
    quote: str = Field(description="The VERBATIM quote from the specified source that justifies the answer.")

class QuotedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""
    answer: str = Field(description="The answer to the user question, which is based only on the given sources. Include any relevant sources in the answer as markdown hyperlinks. For example: 'This is a sample text ([url website](url))'")
    citations: List[Citation] = Field(description="Citations from the given sources that justify the answer.")

class WriteAgent:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o", temperature=0.4, max_tokens=8000)

    def run(self, state: ResearchState, total_words: int = 500):
        # TODO implement a general structure for the report (similar to the way it was done in GPT researcher"
        print(f"✍️ Writing report for '{state['task']['query']}' ...")
        prompt = f"""Today's date is {datetime.now().strftime('%d/%m/%Y')}.\n
        {state['agent']['prompt']}\n
        As an expert researcher, your current task is to write a report on the following query: {state['task']['query']}
        The report should focus on the answer to the query, should be well structured, informative, 
        in-depth, and comprehensive, with facts and numbers if available and at least {total_words} words.\n
        Below are the documents you should base your answer on:\n{state['curated_data']}
        """

        messages = [SystemMessage(content=prompt)]
        response = self.model.with_structured_output(QuotedAnswer).invoke(messages)
        full_report = response.answer
        # Add Citations Section to the report and save sources
        full_report += "\n\n### Citations\n"
        # sources = {}
        for citation in response.citations:
            doc = state['curated_data'].get(citation.source_id)
            full_report += f"- [{doc.get('title', citation.source_id)}]({citation.source_id}): \"{citation.quote}\"\n"
        print("Genereated report:\n",full_report)
        return {"report": full_report}
