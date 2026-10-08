
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.database_models import DatasetRecord

router = APIRouter(prefix="/api/datasets", tags=["Datasets"])

# Temporary compatibility during the migration.
datasets = {}


class DatasetCreate(BaseModel):
    name: str
    format: str
    purpose: str
    description: Optional[str] = None
    size: Optional[int] = None


def serialize_dataset(record):
    return {
        "id": record.id,
        "name": record.name,
        "format": record.format,
        "purpose": record.purpose,
        "description": record.description,
        "size": record.size,
        "file_path": record.file_path,
        "created_at": record.created_at,
    }


@router.post("/")
def create_dataset(data: DatasetCreate, db: Session = Depends(get_db)):
    record = DatasetRecord(
        id=str(uuid4()),
        name=data.name,
        format=data.format,
        purpose=data.purpose,
        description=data.description,
        size=data.size,
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    result = serialize_dataset(record)
    datasets[result["id"]] = result
    return result


@router.get("/")
def list_datasets(db: Session = Depends(get_db)):
    records = db.query(DatasetRecord).all()
    results = [serialize_dataset(record) for record in records]

    datasets.clear()
    datasets.update({item["id"]: item for item in results})

    return results


@router.get("/{dataset_id}")
def read_dataset(dataset_id: str, db: Session = Depends(get_db)):
    record = db.get(DatasetRecord, dataset_id)

    if record is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    result = serialize_dataset(record)
    datasets[dataset_id] = result
    return result


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    record = db.get(DatasetRecord, dataset_id)

    if record is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    db.delete(record)
    db.commit()
    datasets.pop(dataset_id, None)

    return {"message": "Dataset deleted", "dataset_id": dataset_id}
