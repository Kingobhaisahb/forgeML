import math
from datetime import datetime
from pathlib import Path
from threading import Thread

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.database import SessionLocal
from app.database_models import EvaluationRecord, ModelRecord, DatasetRecord
from app.services.artifact_service import get_artifact


BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_DIR / "data"


def evaluation_to_dict(record: EvaluationRecord) -> dict:
    return {
        "id": record.id,
        "artifact_id": record.artifact_id,
        "dataset_id": record.dataset_id,
        "max_samples": record.max_samples,
        "description": record.description,
        "status": record.status,
        "progress": record.progress,
        "current_stage": record.current_stage,
        "error": record.error,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "completed_at": record.completed_at,
        "metrics": record.metrics,
    }


def create_evaluation(
    artifact_id: str,
    dataset_id: str,
    max_samples: int = 1,
    description: str | None = None,
):
    if max_samples < 1:
        raise ValueError("max_samples must be at least 1")

    evaluation_id = __import__("uuid").uuid4().__str__()
    record = EvaluationRecord(
        id=evaluation_id,
        artifact_id=artifact_id,
        dataset_id=dataset_id,
        max_samples=max_samples,
        description=description,
        status="queued",
        progress=0,
        current_stage="queued",
        created_at=datetime.utcnow().isoformat(),
    )

    db = SessionLocal()
    try:
        db.add(record)
        db.commit()
        db.refresh(record)
        response = evaluation_to_dict(record)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    thread = Thread(target=execute_evaluation, args=(evaluation_id,), daemon=True)
    thread.start()
    return response


def execute_evaluation(evaluation_id: str):
    db = SessionLocal()
    try:
        record = db.get(EvaluationRecord, evaluation_id)
        if record is None:
            return
        evaluation = evaluation_to_dict(record)
    finally:
        db.close()

    try:
        update_evaluation(
            evaluation_id,
            started_at=datetime.utcnow().isoformat(),
            status="running",
            progress=5,
            current_stage="loading_artifact",
        )

        artifact = get_artifact(evaluation["artifact_id"])
        if artifact is None:
            raise ValueError(f"Artifact not found: {evaluation['artifact_id']}")

        db = SessionLocal()
        try:
            model_record = db.get(ModelRecord, artifact["model_id"])
            dataset_record = db.get(DatasetRecord, evaluation["dataset_id"])
            model_name = model_record.model_id if model_record else None
            dataset_file_path = dataset_record.file_path if dataset_record else None
        finally:
            db.close()

        if not model_name:
            raise ValueError(f"Model not found: {artifact['model_id']}")
        if dataset_record is None:
            raise ValueError(f"Dataset not found: {evaluation['dataset_id']}")

        artifact_path = Path(artifact["path"])
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact path does not exist: {artifact_path}")

        update_evaluation(
            evaluation_id,
            progress=20,
            current_stage="loading_model",
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(artifact_path))
        model.eval()

        update_evaluation(
            evaluation_id,
            progress=45,
            current_stage="loading_dataset",
        )

        # Prefer the selected dataset's configured file. Existing records without
        # a file_path use the project's local_train.jsonl as a compatibility fallback.
        dataset_path = Path(dataset_file_path) if dataset_file_path else DATA_DIR / "local_train.jsonl"
        if not dataset_path.is_absolute():
            dataset_path = BACKEND_DIR / dataset_path

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

        dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        sample_count = min(evaluation["max_samples"], len(dataset))
        if sample_count <= 0:
            raise ValueError("Evaluation dataset is empty.")

        dataset = dataset.select(range(sample_count))
        update_evaluation(
            evaluation_id,
            progress=60,
            current_stage="evaluating",
        )

        losses = []
        for example in dataset:
            if "instruction" not in example or "response" not in example:
                raise ValueError("Dataset examples must contain 'instruction' and 'response' fields.")

            text = (
                "### Instruction:\n"
                f"{example['instruction']}\n\n"
                "### Response:\n"
                f"{example['response']}"
                f"{tokenizer.eos_token or ''}"
            )
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            losses.append(float(outputs.loss.item()))

        eval_loss = sum(losses) / len(losses)
        perplexity = math.exp(min(eval_loss, 20))
        metrics = {
            "eval_loss": eval_loss,
            "perplexity": perplexity,
            "samples_evaluated": sample_count,
            "artifact_id": evaluation["artifact_id"],
        }

        update_evaluation(
            evaluation_id,
            status="completed",
            progress=100,
            current_stage="completed",
            completed_at=datetime.utcnow().isoformat(),
            error=None,
            metrics=metrics,
        )
    except Exception as exc:
        update_evaluation(
            evaluation_id,
            status="failed",
            current_stage="failed",
            completed_at=datetime.utcnow().isoformat(),
            error=str(exc),
        )


def update_evaluation(evaluation_id: str, **updates):
    db = SessionLocal()
    try:
        record = db.get(EvaluationRecord, evaluation_id)
        if record is None:
            return
        for key, value in updates.items():
            if hasattr(record, key):
                setattr(record, key, value)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_evaluation(evaluation_id: str):
    db = SessionLocal()
    try:
        record = db.get(EvaluationRecord, evaluation_id)
        return evaluation_to_dict(record) if record else None
    finally:
        db.close()


def get_evaluations():
    db = SessionLocal()
    try:
        records = (
            db.query(EvaluationRecord)
            .order_by(EvaluationRecord.created_at.desc())
            .all()
        )
        return [evaluation_to_dict(record) for record in records]
    finally:
        db.close()


def delete_evaluation(evaluation_id: str):
    db = SessionLocal()
    try:
        record = db.get(EvaluationRecord, evaluation_id)
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
