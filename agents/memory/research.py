from typing import TypedDict, Dict, Union, List, Annotated, Required

class ResearchState(TypedDict):
    task: Required[dict]
    agent: dict
    research_data: Dict[str, Dict[str, Union[str, float]]]
    curated_data: Dict[str, Dict[str, Union[str, float]]]
    report: str
