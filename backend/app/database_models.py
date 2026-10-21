
from datetime import datetime

from sqlalchemy import Column, String, Integer, Float, JSON, Text

from app.database import Base


def timestamp():
    return datetime.utcnow().isoformat()


class ModelRecord(Base):
    __tablename__ = "models"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    model_id = Column(String, nullable=False)
    description = Column(String, nullable=True)
    created_at = Column(String, default=timestamp)


class DatasetRecord(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    format = Column(String, nullable=False)
    purpose = Column(String, nullable=False)
    description = Column(String, nullable=True)
    size = Column(Integer, nullable=True)
    file_path = Column(String, nullable=True)
    created_at = Column(String, default=timestamp)


class TrainingJobRecord(Base):
    __tablename__ = "training_jobs"

    id = Column(String, primary_key=True)
    model_id = Column(String, nullable=False)
    dataset_id = Column(String, nullable=False)
    method = Column(String, nullable=False)
    epochs = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    batch_size = Column(Integer, nullable=False)
    description = Column(String, nullable=True)
    status = Column(String, default="queued")
    progress = Column(Integer, default=0)
    current_stage = Column(String, default="queued")
    error = Column(String, nullable=True)
    created_at = Column(String, default=timestamp)
    started_at = Column(String, nullable=True)
    completed_at = Column(String, nullable=True)
    metrics = Column(JSON, nullable=True)
    artifact_path = Column(String, nullable=True)
    artifact_id = Column(String, nullable=True)


class ArtifactRecord(Base):
    __tablename__ = "artifacts"

    id = Column(String, primary_key=True)
    training_job_id = Column(String, nullable=False)
    model_id = Column(String, nullable=False)
    dataset_id = Column(String, nullable=False)
    method = Column(String, nullable=False)
    type = Column(String, default="lora_adapter")
    path = Column(String, nullable=False)
    metrics = Column(JSON, nullable=True)
    created_at = Column(String, default=timestamp)


class EvaluationRecord(Base):
    __tablename__ = "evaluations"

    id = Column(String, primary_key=True)
    artifact_id = Column(String, nullable=False)
    dataset_id = Column(String, nullable=False)
    max_samples = Column(Integer, default=1)
    description = Column(String, nullable=True)
    status = Column(String, default="queued")
    progress = Column(Integer, default=0)
    current_stage = Column(String, default="queued")
    error = Column(String, nullable=True)
    created_at = Column(String, default=timestamp)
    started_at = Column(String, nullable=True)
    completed_at = Column(String, nullable=True)
    metrics = Column(JSON, nullable=True)


class InferenceRecord(Base):
    __tablename__ = "inference_history"

    id = Column(String, primary_key=True, index=True)
    artifact_id = Column(String, nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    max_new_tokens = Column(Integer, nullable=False)
    temperature = Column(Float, nullable=False)
    generation_time_seconds = Column(Float, nullable=False)
    prompt_tokens = Column(Integer, nullable=False)
    generated_tokens = Column(Integer, nullable=False)
    created_at = Column(String, default=timestamp, nullable=False)


class ExperimentRecord(Base):
    __tablename__ = "experiments"

    id = Column(String, primary_key=True, index=True)
    artifact_id = Column(String, nullable=False, index=True)

    prompt = Column(Text, nullable=False)
    max_new_tokens = Column(Integer, nullable=False)
    temperature = Column(Float, nullable=False)

    base_model_name = Column(String, nullable=False)
    fine_tuned_model_name = Column(String, nullable=False)

    base_response = Column(Text, nullable=False)
    fine_tuned_response = Column(Text, nullable=False)

    base_generation_time_seconds = Column(Float, nullable=False)
    fine_tuned_generation_time_seconds = Column(Float, nullable=False)

    base_prompt_tokens = Column(Integer, nullable=False)
    fine_tuned_prompt_tokens = Column(Integer, nullable=False)

    base_generated_tokens = Column(Integer, nullable=False)
    fine_tuned_generated_tokens = Column(Integer, nullable=False)

    created_at = Column(String, default=timestamp, nullable=False)
