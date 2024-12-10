from pydantic import BaseModel, Field
from typing import TypedDict, Dict, Union, List, Annotated, Required, NotRequired, Literal
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

class InputState(BaseModel):
    query: str = Field(
        description="The research query to be used for generating the report.",
        examples=["What are the benefits of renewable energy?", "Impact of climate change on biodiversity"],
    )
    research_depth: Literal["basic", "advanced"] = Field(
        default="basic",
        description="The research depth of the report. 'basic' provides an overview, while 'advanced' dives deeper into the topic. Default is 'basic'",
        examples=["basic", "advanced"],
    )
    include_citations: bool = Field(
        default=False,
        description="Whether to include cited sources and supporting quotes from those sources at the end of the report. Default is False.",
        examples=[True, False],
    )

class OutputState(BaseModel):
    report: str = ""

class ResearchState(InputState, OutputState):
    agent: dict = Field(default_factory=dict)
    research_data: Dict[str, Dict[str, Union[str, float, None]]] = Field(default_factory=dict)
    curated_data: Dict[str, Dict[str, Union[str, float, None]]] = Field(default_factory=dict)
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)

