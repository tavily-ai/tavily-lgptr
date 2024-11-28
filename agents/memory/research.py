from pydantic import BaseModel, Field
from typing import TypedDict, Dict, Union, List, Annotated, Required, NotRequired, Literal
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


# class InputState(TypedDict):
#     query: str
#     research_depth: NotRequired[str]
#     include_citations: bool
#
#
# class OutputState(TypedDict):
#     report: str
#
#
# class ResearchState(InputState, OutputState):
#     agent: dict
#     research_data: Dict[str, Dict[str, Union[str, float]]]
#     curated_data: Dict[str, Dict[str, Union[str, float]]]
#     messages: Annotated[list[AnyMessage], add_messages]  # TODO: use this field for streaming msgs from agent?

class InputState(BaseModel):
    query: str
    research_depth: Literal["basic", "advanced"] = "basic"  # Enforce specific values
    include_citations: bool = False

class OutputState(BaseModel):
    report: str = ""


class ResearchState(InputState, OutputState):
    agent: dict = Field(default_factory=dict)
    research_data: Dict[str, Dict[str, Union[str, float, None]]] = Field(default_factory=dict)
    curated_data: Dict[str, Dict[str, Union[str, float, None]]] = Field(default_factory=dict)
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)

