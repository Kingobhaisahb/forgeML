
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.database_models import ModelRecord

router = APIRouter(prefix="/api/models", tags=["Models"])

# Temporary compatibility for existing services that import this dictionary.
models = {}


class ModelCreate(BaseModel):
    name: str
    provider: str
    model_id: str
    description: Optional[str] = None


def serialize_model(record):
    return {
        "id": record.id,
        "name": record.name,
        "provider": record.provider,
        "model_id": record.model_id,
        "description": record.description,
        "created_at": record.created_at,
    }


@router.post("/")
def create_model(data: ModelCreate, db: Session = Depends(get_db)):
    record = ModelRecord(
        id=str(uuid4()),
        name=data.name,
        provider=data.provider,
        model_id=data.model_id,
        description=data.description,
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    result = serialize_model(record)
    models[result["id"]] = result
    return result


@router.get("/")
def list_models(db: Session = Depends(get_db)):
    records = db.query(ModelRecord).all()
    results = [serialize_model(record) for record in records]

    # Keep existing services compatible during migration.
    models.clear()
    models.update({item["id"]: item for item in results})

    return results


@router.get("/{model_id}")
def read_model(model_id: str, db: Session = Depends(get_db)):
    record = db.get(ModelRecord, model_id)

    if record is None:
        raise HTTPException(status_code=404, detail="Model not found")

    result = serialize_model(record)
    models[model_id] = result
    return result


@router.delete("/{model_id}")
def delete_model(model_id: str, db: Session = Depends(get_db)):
    record = db.get(ModelRecord, model_id)

    if record is None:
        raise HTTPException(status_code=404, detail="Model not found")

    db.delete(record)
    db.commit()
    models.pop(model_id, None)

    return {"message": "Model deleted", "model_id": model_id}
