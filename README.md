# GPT Researcher powered by Tavily

This project implements GPT Researcher powered by Tavily's search and extract API. It leverages LangGraph that runs a combination of AI agents to generate, search, curate, and write research reports based on user queries.

## Installation

1. Clone the repository
2. Install the required packages: `pip install -r requirements.txt`
3. Create a `.env` file in the project root and add your API keys:   ```
   OPENAI_API_KEY=your_openai_api_key_here 
   TAVILY_API_KEY=your_tavily_api_key_here   ```

## Usage

Run the main script from the project root: `python main.py`