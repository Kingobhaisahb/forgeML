import math
from datetime import datetime
from pathlib import Path
from threading import Thread
from uuid import uuid4

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from app.api.models import models
from app.services.artifact_service import get_artifact


evaluations = {}

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_DIR / "data"


def create_evaluation(
    artifact_id: str,
    dataset_id: str,
    max_samples: int = 1,
    description: str | None = None,
):
    evaluation_id = str(uuid4())

    evaluation = {
        "id": evaluation_id,
        "artifact_id": artifact_id,
        "dataset_id": dataset_id,
        "max_samples": max_samples,
        "description": description,

        "status": "queued",
        "progress": 0,
        "current_stage": "queued",

        "error": None,

        "created_at": datetime.utcnow().isoformat(),
        "started_at": None,
        "completed_at": None,

        "metrics": None,
    }

    evaluations[evaluation_id] = evaluation

    thread = Thread(
        target=execute_evaluation,
        args=(evaluation_id,),
        daemon=True,
    )

    thread.start()

    return evaluation


def execute_evaluation(evaluation_id: str):

    evaluation = evaluations.get(evaluation_id)

    if evaluation is None:
        return

    try:
        evaluation["started_at"] = datetime.utcnow().isoformat()

        update_evaluation(
            evaluation_id,
            status="running",
            progress=10,
            current_stage="loading_artifact",
        )

        artifact = get_artifact(
            evaluation["artifact_id"]
        )

        if artifact is None:
            raise ValueError(
                f"Artifact not found: {evaluation['artifact_id']}"
            )

        model_record = models.get(
            artifact["model_id"]
        )

        if model_record is None:
            raise ValueError(
                f"Model not found: {artifact['model_id']}"
            )

        model_name = model_record["model_id"]

        artifact_path = Path(
            artifact["path"]
        )

        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Artifact path does not exist: {artifact_path}"
            )

        update_evaluation(
            evaluation_id,
            progress=25,
            current_stage="loading_model",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        model = PeftModel.from_pretrained(
            base_model,
            str(artifact_path),
        )

        model.eval()

        update_evaluation(
            evaluation_id,
            progress=50,
            current_stage="loading_dataset",
        )

        dataset_path = DATA_DIR / "local_train.jsonl"

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}"
            )

        dataset = load_dataset(
            "json",
            data_files=str(dataset_path),
            split="train",
        )

        max_samples = min(
            evaluation["max_samples"],
            len(dataset),
        )

        if max_samples <= 0:
            raise ValueError(
                "Evaluation dataset is empty."
            )

        dataset = dataset.select(
            range(max_samples)
        )

        update_evaluation(
            evaluation_id,
            progress=65,
            current_stage="evaluating",
        )

        losses = []

        for example in dataset:

            text = (
                "### Instruction:\n"
                f"{example['instruction']}\n\n"
                "### Response:\n"
                f"{example['response']}"
                f"{tokenizer.eos_token}"
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

            labels[
                attention_mask == 0
            ] = -100

            with torch.no_grad():

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            losses.append(
                float(outputs.loss.item())
            )

        eval_loss = sum(losses) / len(losses)

        perplexity = math.exp(
            min(eval_loss, 20)
        )

        metrics = {
            "eval_loss": eval_loss,
            "perplexity": perplexity,
            "samples_evaluated": max_samples,
            "artifact_id": evaluation["artifact_id"],
        }

        update_evaluation(
            evaluation_id,

            status="completed",

            progress=100,

            current_stage="completed",

            completed_at=datetime.utcnow().isoformat(),

            metrics=metrics,
        )

    except Exception as exc:

        update_evaluation(
            evaluation_id,

            status="failed",

            current_stage="failed",

            error=str(exc),
        )


def update_evaluation(
    evaluation_id: str,
    **updates,
):
    if evaluation_id not in evaluations:
        return

    evaluations[evaluation_id].update(
        updates
    )


def get_evaluation(evaluation_id: str):
    return evaluations.get(
        evaluation_id
    )


def get_evaluations():
    return list(
        evaluations.values()
    )


def delete_evaluation(evaluation_id: str):

    if evaluation_id not in evaluations:
        return False

    del evaluations[evaluation_id]

    return True