from pydantic import BaseModel, Field, root_validator
from typing import Optional, List


class OutputParser(BaseModel):
    rewritten_query: str = Field(..., description="Rewritten positive query or query with exclusion criteria removed if rewriting is not possible due to information loss")
    exclusion_criteria: List[str] = Field([], description="List of terms or phrases to exclude from results")
    explanation: str = Field("", description="Brief explanation of rewrite or filtering decision")

    @classmethod
    def from_llm_response(cls, llm_output: dict) -> "OutputParser":
        # Example llm_output:
        # {
        #   "rewritten_query": "Films about World War III",
        #   "exclusion_criteria": ["Environmental films"],
        #   "explanation": "Removed negation phrase; exclusion criteria specified for post-retrieval filtering."
        # }
        return cls.parse_obj(llm_output)
