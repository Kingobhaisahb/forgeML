from pydantic import BaseModel, Field
from typing import Literal, Optional


class TrainingJobCreate(BaseModel):
    model_id: str
    dataset_id: str

    method: Literal["lora", "full_finetuning"] = "lora"

    epochs: int = Field(default=3, ge=1, le=100)
    learning_rate: float = Field(default=0.0002, gt=0)
    batch_size: int = Field(default=2, ge=1)

    description: Optional[str] = None


class TrainingJobResponse(BaseModel):
    id: str

    model_id: str
    dataset_id: str

    method: str

    epochs: int
    learning_rate: float
    batch_size: int

    description: Optional[str] = None

    status: str