import asyncio
from agents.master import MasterAgent

from dotenv import load_dotenv

load_dotenv('.env')


async def main():
    master_agent = MasterAgent()
    query = input("What do you want to research?\n")
    await master_agent.run(query)


if __name__ == "__main__":
    asyncio.run(main())
