# Description: Configuration file
class Config:
    def __init__(self):
        """
        Initializes the configuration for the agents
        """
        self.MAX_SEARCH_QUERIES = 4  # Max search queries to run (including initial search query)
        self.MAX_CURATED_DOCS = 10
        self.RESEARCH_DEPTH = "basic"  # "basic" or "advanced"

