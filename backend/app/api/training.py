from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.services.training_service import (
    create_training_job,
    get_training_job,
    get_training_jobs,
    delete_training_job,
)

from typing import Optional, Literal


router = APIRouter(
    prefix="/api/training-jobs",
    tags=["Training Jobs"],
)


class TrainingJobCreate(BaseModel):
    model_id: str
    dataset_id: str
    method: str
    epochs: int
    learning_rate: float
    batch_size: int
    description: Optional[str] = None
    compute_target: Literal["local", "amd_mi300x"] = "local"


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
    progress: int
    current_stage: str

    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    metrics: Optional[dict] = None
    artifact_path: Optional[str] = None

    compute_target: str = "local"
    artifact_id: Optional[str] = None


@router.post(
    "/",
    response_model=TrainingJobResponse,
)
def create_job(job: TrainingJobCreate):

    return create_training_job(
        job.model_dump()
    )


@router.get(
    "/",
    response_model=list[TrainingJobResponse],
)
def list_jobs():

    return get_training_jobs()


@router.get(
    "/{job_id}",
    response_model=TrainingJobResponse,
)
def get_job(job_id: str):

    job = get_training_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Training job not found",
        )

    return job


@router.delete(
    "/{job_id}",
)
def delete_job(job_id: str):

    deleted = delete_training_job(job_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Training job not found",
        )

    return {
        "message": "Training job deleted successfully",
        "id": job_id,
    }