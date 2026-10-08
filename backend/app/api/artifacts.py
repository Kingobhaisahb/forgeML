from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.artifact_service import (
    create_artifact,
    get_artifact,
    get_artifacts,
    delete_artifact,
)

router = APIRouter(
    prefix="/api/artifacts",
    tags=["Artifacts"],
)


class ArtifactCreate(BaseModel):
    training_job_id: str
    model_id: str
    dataset_id: str
    method: str = "lora"
    artifact_path: str
    metrics: dict | None = None


@router.post("/")
def register_artifact(data: ArtifactCreate):
    artifact = create_artifact(
        training_job_id=data.training_job_id,
        model_id=data.model_id,
        dataset_id=data.dataset_id,
        method=data.method,
        artifact_path=data.artifact_path,
        metrics=data.metrics,
    )

    return artifact


@router.get("/")
def list_artifacts():
    return get_artifacts()


@router.get("/{artifact_id}")
def read_artifact(artifact_id: str):
    artifact = get_artifact(artifact_id)

    if artifact is None:
        raise HTTPException(
            status_code=404,
            detail="Artifact not found",
        )

    return artifact


@router.delete("/{artifact_id}")
def remove_artifact(artifact_id: str):
    deleted = delete_artifact(artifact_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Artifact not found",
        )

    return {
        "message": "Artifact deleted",
        "artifact_id": artifact_id,
    }