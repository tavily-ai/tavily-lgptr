from typing import TypedDict, Dict, Union, List, Annotated, Required, NotRequired
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class InputState(TypedDict):
    query: str
    include_citations: NotRequired[bool]


class OutputState(TypedDict):
    report: str


class ResearchState(InputState, OutputState):
    agent: dict
    research_data: Dict[str, Dict[str, Union[str, float]]]
    curated_data: Dict[str, Dict[str, Union[str, float]]]
    messages: Annotated[list[AnyMessage], add_messages]  # TODO: use this field for streaming msgs from agent?
