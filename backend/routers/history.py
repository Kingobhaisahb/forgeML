from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.database import SessionLocal
from app.database_models import InferenceRecord

router = APIRouter(prefix="/api/inference-history", tags=["Inference History"])


class SaveInferenceRequest(BaseModel):
    artifact_id: str
    prompt: str
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
                "id": r.id,
                "artifact_id": r.artifact_id,
                "prompt": r.prompt,
                "response": r.response,
                "max_new_tokens": r.max_new_tokens,
                "temperature": r.temperature,
                "generation_time_seconds": r.generation_time_seconds,
                "prompt_tokens": r.prompt_tokens,
                "generated_tokens": r.generated_tokens,
                "created_at": r.created_at,
            }
            for r in records
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
        return {"id": record.id, "status": "saved"}
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        db.close()


@router.delete("/{record_id}")
def delete_history(record_id: str):
    db = SessionLocal()
    try:
        record = db.get(InferenceRecord, record_id)
        if record is None:
            raise HTTPException(status_code=404, detail="History record not found")
        db.delete(record)
        db.commit()
        return {"status": "deleted", "id": record_id}
    finally:
        db.close()