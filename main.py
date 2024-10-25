import asyncio
from agents.master import MasterAgent

from dotenv import load_dotenv
load_dotenv('.env.local')

async def main():
    master_agent = MasterAgent()
    query = input("What do you want to research?\n")
    task = {
        "query": query
    }
    await master_agent.run(task)

if __name__ == "__main__":
    asyncio.run(main())
