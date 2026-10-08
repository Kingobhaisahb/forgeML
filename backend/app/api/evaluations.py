from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.evaluation_service import (
    create_evaluation,
    get_evaluation,
    get_evaluations,
    delete_evaluation,
)


router = APIRouter(
    prefix="/api/evaluations",
    tags=["Evaluations"],
)


class EvaluationCreate(BaseModel):
    artifact_id: str
    dataset_id: str
    max_samples: int = 1
    description: str | None = None


@router.post("/")
def start_evaluation(
    data: EvaluationCreate,
):
    return create_evaluation(
        artifact_id=data.artifact_id,
        dataset_id=data.dataset_id,
        max_samples=data.max_samples,
        description=data.description,
    )


@router.get("/")
def list_evaluations():
    return get_evaluations()


@router.get("/{evaluation_id}")
def read_evaluation(
    evaluation_id: str,
):
    evaluation = get_evaluation(
        evaluation_id
    )

    if evaluation is None:
        raise HTTPException(
            status_code=404,
            detail="Evaluation not found",
        )

    return evaluation


@router.delete("/{evaluation_id}")
def remove_evaluation(
    evaluation_id: str,
):
    deleted = delete_evaluation(
        evaluation_id
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Evaluation not found",
        )

    return {
        "message": "Evaluation deleted",
        "evaluation_id": evaluation_id,
    }