from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class WriteAgent:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)