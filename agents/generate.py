from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

from memory.research import ResearchState


class GeneratorResponse(BaseModel):
    agent_name: str
    agent_prompt: str


class GenerateAgent:
    """Agent responsible for generating an agent based on research task"""

    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.system_prompt = """
This task involves researching a given topic, regardless of its complexity or the availability of a definitive answer. The research is conducted by a specific agent, defined by its type and role, with each agent requiring distinct instructions.
The server is determined by the field of the topic and the specific name of the agent that could be utilized to research the topic provided. Agents are categorized by their area of expertise, and each agent type is associated with a corresponding emoji.

examples:
task: "should I invest in apple stocks?"
response: 
{
    "agent_name": "💰 Finance Agent",
    "agent_prompt: "You are a seasoned finance analyst AI assistant. Your primary goal is to compose comprehensive, astute, impartial, and methodically arranged financial reports based on provided data and trends."
}
task: "could reselling sneakers become profitable?"
response: 
{ 
    "agent_name":  "📈 Business Analyst Agent",
    "agent_prompt": "You are an experienced AI business analyst assistant. Your main objective is to produce comprehensive, insightful, impartial, and systematically structured business reports based on provided business data, market trends, and strategic analysis."
}
task: "what are the most interesting sites in Tel Aviv?"
response:
{
    "agent_name:  "🌍 Travel Agent",
    "agent_prompt": "You are a world-travelled AI tour guide assistant. Your main purpose is to draft engaging, insightful, unbiased, and well-structured travel reports on given locations, including history, attractions, and cultural insights."
}
"""

    async def run(self, state: ResearchState):
        """
        Chooses the agent automatically
        Args:
            query: original query
        Returns:
            agent: Agent name
            agent_role_prompt: Agent role prompt
        """
        query = state['task']['query']
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=query)]
        try:
            response = await self.model.with_structured_output(GeneratorResponse).ainvoke(messages)
            print(f"Generated {response.agent_name}")
            return {
                "agent":
                    {
                        "name": response.agent_name,
                        "prompt": response.agent_prompt
                    }
            }
        except Exception as e:
            print("⚠️ Error in reading GeneratorAgent response, returning Default Agent")
            return {
                "agent":
                    {
                        "name": "Default Agent",
                        "prompt": """You are an AI critical thinker research assistant.
                                  "Your sole purpose is to write well written,critically acclaimed, objective and structured reports on given text."""
                    }
            }
