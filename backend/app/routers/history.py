
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.database import SessionLocal
from app.database_models import InferenceRecord

router = APIRouter(
    prefix="/api/inference-history",
    tags=["Inference History"],
)


class SaveInferenceRequest(BaseModel):
    artifact_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    response: str
    max_new_tokens: int = Field(ge=1, le=512)
    temperature: float = Field(ge=0, le=2)
    generation_time_seconds: float = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    generated_tokens: int = Field(ge=0)


@router.get("/")
def list_history(limit: int = 50):
    limit = max(1, min(limit, 200))

    db = SessionLocal()

    try:
        records = (
            db.query(InferenceRecord)
            .order_by(InferenceRecord.created_at.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "id": record.id,
                "artifact_id": record.artifact_id,
                "prompt": record.prompt,
                "response": record.response,
                "max_new_tokens": record.max_new_tokens,
                "temperature": record.temperature,
                "generation_time_seconds": (
                    record.generation_time_seconds
                ),
                "prompt_tokens": record.prompt_tokens,
                "generated_tokens": record.generated_tokens,
                "created_at": record.created_at,
            }
            for record in records
        ]

    finally:
        db.close()


@router.post("/")
def save_history(request: SaveInferenceRequest):
    db = SessionLocal()

    try:
        record = InferenceRecord(
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
            detail=f"Could not save inference history: {exc}",
        ) from exc

    finally:
        db.close()


@router.delete("/{record_id}")
def delete_history(record_id: str):
    db = SessionLocal()

    try:
        record = db.get(InferenceRecord, record_id)

        if record is None:
            raise HTTPException(
                status_code=404,
                detail="History record not found",
            )

        db.delete(record)
        db.commit()

        return {
            "status": "deleted",
            "id": record_id,
        }

    finally:
        db.close()
