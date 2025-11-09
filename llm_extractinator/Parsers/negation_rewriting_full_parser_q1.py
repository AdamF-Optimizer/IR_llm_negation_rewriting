from pydantic import BaseModel, Field, root_validator
from typing import Optional, List

class NegationType(BaseModel):
    name: str = Field(..., description="Name of the negation type as per taxonomy")
    span_start: Optional[int] = Field(None, description="Start index of negation phrase span")
    span_end: Optional[int] = Field(None, description="End index of negation phrase span")
    confidence: Optional[float] = Field(None, description="Confidence score of detection")

class OutputParser(BaseModel):
    negation_types_q1: List[NegationType] = Field(default_factory=list, description="List of detected negation types in the text")
    is_negated_q1: bool = Field(..., description="Whether negation occurs in the text based on the taxonomy")
    remarks_q1: str = Field("", description="remarks or comments about the negation detection")
    rewritten_query_q1: str = Field(..., description="Rewritten positive query or query with exclusion criteria removed if rewriting is not possible due to information loss")
    exclusion_criteria_q1: List[str] = Field([], description="List of terms or phrases to exclude from results")
    explanation_q1: str = Field("", description="Brief explanation of rewrite or filtering decision")

    @classmethod
    def from_llm_response(cls, llm_output: dict) -> "OutputParser":
        # Example llm_output:
        # Expected llm_output format example:
        # {
        #   "negation_types": [
        #      {"name": "sentential negation", "span_start": 4, "span_end": 7, "confidence": 0.95},
        #      {"name": "exclusionary negation", "span_start": null, "span_end": null, "confidence": 0.88}
        #    ],
        #   "is_negated": true,
        #   "remarks": "detected multiple negation forms, form 1, form 2 etc",
        #   "rewritten_query": "Films about World War III",
        #   "exclusion_criteria": ["Environmental films"],
        #   "explanation": "Removed negation phrase; exclusion criteria specified for post-retrieval filtering."
        # }
        return cls.parse_obj(llm_output)
