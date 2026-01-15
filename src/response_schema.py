from pydantic import BaseModel
from typing import List, Literal

# Sub-model for the items inside the 'artifacts' array
class Artifact(BaseModel):
    type: Literal["structural", "physical", "semantic", "stylistic"]
    description: str

# Main model for the API response
class givenTypes(BaseModel):
    classification: Literal["real", "fake"]
    artifacts: List[Artifact]


# Main model for the API response
class explain(BaseModel):
    classification: Literal["real", "fake"]
    explanation: str