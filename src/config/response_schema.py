from pydantic import BaseModel
from typing import List, Literal


# Schema for Prompt 1
class classification(BaseModel):
    classification: Literal["real", "fake"]

class ArtifactTypes(BaseModel):
    type: Literal["structural", "physical", "semantic", "stylistic"]

# Schema for Prompt 2
class artifacts(BaseModel):
    classification: Literal["real", "fake"]
    artifacts: List[ArtifactTypes]

# Schema for Prompt 3
class ArtifactExplanation(BaseModel):
    type: Literal["structural", "physical", "semantic", "stylistic"]
    reasoning: str
    location: str

class reasoning(BaseModel):
    classification: Literal["real", "fake"]
    artifacts: List[ArtifactExplanation]


# Schema for Prompt 4
class explainArtifact(BaseModel):
    reasoning: str
    location: str

class explain(BaseModel):
    classification: Literal["real", "fake"]
    artifacts: List[explainArtifact]