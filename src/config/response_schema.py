from pydantic import BaseModel
from typing import List, Literal

# Sub-model for the items inside the 'artifacts' array
class ArtifactTypes(BaseModel):
    type: Literal["structural", "physical", "semantic", "stylistic"]

# Main model for the API response
class artifacts(BaseModel):
    classification: Literal["real", "fake"]
    artifacts: List[ArtifactTypes]


# Main model for the API response
class explain(BaseModel):
    classification: Literal["real", "fake"]
    explanation: str