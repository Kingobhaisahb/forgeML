
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.database import SessionLocal
from app.database_models import ExperimentRecord

router = APIRouter(
    prefix="/api/experiments",
    tags=["Experiments"],
)


class SaveExperimentRequest(BaseModel):
    artifact_id: str = Field(min_length=1)

    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(ge=1, le=512)
    temperature: float = Field(ge=0, le=2)

    base_model_name: str
    fine_tuned_model_name: str

    base_response: str
    fine_tuned_response: str

    base_generation_time_seconds: float = Field(ge=0)
    fine_tuned_generation_time_seconds: float = Field(ge=0)

    base_prompt_tokens: int = Field(ge=0)
    fine_tuned_prompt_tokens: int = Field(ge=0)

    base_generated_tokens: int = Field(ge=0)
    fine_tuned_generated_tokens: int = Field(ge=0)


def experiment_to_dict(record: ExperimentRecord):
    return {
        "id": record.id,
        "artifact_id": record.artifact_id,
        "prompt": record.prompt,
        "max_new_tokens": record.max_new_tokens,
        "temperature": record.temperature,
        "base_model_name": record.base_model_name,
        "fine_tuned_model_name": record.fine_tuned_model_name,
        "base_response": record.base_response,
        "fine_tuned_response": record.fine_tuned_response,
        "base_generation_time_seconds": (
            record.base_generation_time_seconds
        ),
        "fine_tuned_generation_time_seconds": (
            record.fine_tuned_generation_time_seconds
        ),
        "base_prompt_tokens": record.base_prompt_tokens,
        "fine_tuned_prompt_tokens": record.fine_tuned_prompt_tokens,
        "base_generated_tokens": record.base_generated_tokens,
        "fine_tuned_generated_tokens": (
            record.fine_tuned_generated_tokens
        ),
        "created_at": record.created_at,
    }


@router.get("/")
def list_experiments(limit: int = 50):
    limit = max(1, min(limit, 200))

    db = SessionLocal()

    try:
        records = (
            db.query(ExperimentRecord)
            .order_by(ExperimentRecord.created_at.desc())
            .limit(limit)
            .all()
        )

        return [experiment_to_dict(record) for record in records]

    finally:
        db.close()


@router.get("/{experiment_id}")
def get_experiment(experiment_id: str):
    db = SessionLocal()

    try:
        record = db.get(ExperimentRecord, experiment_id)

        if record is None:
            raise HTTPException(
                status_code=404,
                detail="Experiment not found",
            )

        return experiment_to_dict(record)

    finally:
        db.close()


@router.post("/")
def save_experiment(request: SaveExperimentRequest):
    db = SessionLocal()

    try:
        record = ExperimentRecord(
            id=str(uuid4()),
            **request.model_dump(),
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "id": record.id,
            "status": "saved",
        }

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Could not save experiment: {exc}",
        ) from exc

    finally:
        db.close()


@router.delete("/{experiment_id}")
def delete_experiment(experiment_id: str):
    db = SessionLocal()

    try:
        record = db.get(ExperimentRecord, experiment_id)

        if record is None:
            raise HTTPException(
                status_code=404,
                detail="Experiment not found",
            )

        db.delete(record)
        db.commit()

        return {
            "status": "deleted",
            "id": experiment_id,
        }

    finally:
        db.close()
