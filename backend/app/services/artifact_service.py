from datetime import datetime
from uuid import uuid4

from app.database import SessionLocal
from app.database_models import ArtifactRecord


def artifact_to_dict(record: ArtifactRecord) -> dict:
    """Convert a database artifact record into the API response shape."""
    return {
        "id": record.id,
        "training_job_id": record.training_job_id,
        "model_id": record.model_id,
        "dataset_id": record.dataset_id,
        "method": record.method,
        "type": record.type,
        "path": record.path,
        "metrics": record.metrics,
        "created_at": record.created_at,
    }


def create_artifact(
    training_job_id: str,
    model_id: str,
    dataset_id: str,
    method: str,
    artifact_path: str,
    metrics: dict | None = None,
):
    artifact = ArtifactRecord(
        id=str(uuid4()),
        training_job_id=training_job_id,
        model_id=model_id,
        dataset_id=dataset_id,
        method=method,
        type="lora_adapter",
        path=artifact_path,
        metrics=metrics,
        created_at=datetime.utcnow().isoformat(),
    )

    db = SessionLocal()
    try:
        db.add(artifact)
        db.commit()
        db.refresh(artifact)
        return artifact_to_dict(artifact)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_artifact(artifact_id: str):
    db = SessionLocal()
    try:
        record = db.get(ArtifactRecord, artifact_id)
        return artifact_to_dict(record) if record else None
    finally:
        db.close()


def get_artifacts():
    db = SessionLocal()
    try:
        records = (
            db.query(ArtifactRecord)
            .order_by(ArtifactRecord.created_at.desc())
            .all()
        )
        return [artifact_to_dict(record) for record in records]
    finally:
        db.close()


def delete_artifact(artifact_id: str):
    """Delete artifact metadata only; never delete adapter files from disk."""
    db = SessionLocal()
    try:
        record = db.get(ArtifactRecord, artifact_id)
        if record is None:
            return False
        db.delete(record)
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
