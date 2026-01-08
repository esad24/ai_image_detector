from pydantic import BaseModel
from typing import List, Literal

# Sub-model for the items inside the 'artifacts' array
class Artifact(BaseModel):
    type: Literal["structural", "physics", "semantic", "stylistic"]
    description: str
    location: str

# Main model for the API response
class ImageAuthenticityResult(BaseModel):
    classification: Literal["real", "fake"]
    artifacts: List[Artifact]