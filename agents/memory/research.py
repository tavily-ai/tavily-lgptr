from typing import TypedDict, Dict, Union, List, Annotated

class ResearchState(TypedDict):
    task: dict
    agent: dict
    research_data: Dict[str, Dict[str, Union[str, float]]]
    curated_data: Dict[str, Dict[str, Union[str, float]]]
    report: str
