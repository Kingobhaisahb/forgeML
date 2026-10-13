import os

from datetime import datetime
from pathlib import Path
from threading import Thread
from uuid import uuid4

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from app.services.artifact_service import create_artifact
from app.database import SessionLocal
from app.database_models import (
    ModelRecord,
    DatasetRecord,
    TrainingJobRecord,
)


# -------------------------------------------------------------------
# In-memory training job registry
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_DIR / "data"
OUTPUT_DIR = BACKEND_DIR / "outputs"

# -------------------------------------------------------------------
# Training controls
# -------------------------------------------------------------------

FORGEML_MAX_STEPS = int(
    os.getenv("FORGEML_MAX_STEPS", "-1")
)


# -------------------------------------------------------------------
# Real-time training progress callback
# -------------------------------------------------------------------

class ForgeMLTrainingCallback(TrainerCallback):

    def __init__(self, job_id: str):
        self.job_id = job_id

    def on_train_begin(self, args, state, control, **kwargs):
        update_job(
            self.job_id,
            progress=60,
            current_stage="training",
        )

    def on_log(self, args, state, control, logs=None, **kwargs):

        if not logs:
            return

        # Hugging Face gives us the current training step.
        current_step = state.global_step
        total_steps = state.max_steps

        if total_steps > 0:
            # Training occupies the 60% -> 90% range.
            training_progress = 60 + int(
                (current_step / total_steps) * 30
            )

            training_progress = min(training_progress, 89)
        else:
            training_progress = 60

        updates = {
            "progress": training_progress,
            "current_stage": "training",
        }

        # Capture useful live training information.
        if "loss" in logs:
            updates["metrics"] = {
                "loss": logs["loss"],
                "step": current_step,
                "total_steps": total_steps,
                "epoch": logs.get("epoch"),
                "learning_rate": logs.get("learning_rate"),
            }

        update_job(
            self.job_id,
            **updates,
        )

    def on_train_end(self, args, state, control, **kwargs):
        update_job(
            self.job_id,
            progress=90,
            current_stage="training_completed",
        )


def job_to_dict(record):
    return {
        "id": record.id,
        "model_id": record.model_id,
        "dataset_id": record.dataset_id,
        "method": record.method,
        "epochs": record.epochs,
        "learning_rate": record.learning_rate,
        "batch_size": record.batch_size,
        "max_steps": FORGEML_MAX_STEPS,
        "description": record.description,
        "status": record.status,
        "progress": record.progress,
        "current_stage": record.current_stage,
        "error": record.error,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "completed_at": record.completed_at,
        "metrics": record.metrics,
        "artifact_path": record.artifact_path,
        "artifact_id": record.artifact_id,
    }


def update_job(job_id: str, **updates):
    db = SessionLocal()
    try:
        record = db.get(TrainingJobRecord, job_id)
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


def create_training_job(job_data: dict):
    job_id = str(uuid4())
    now = datetime.utcnow().isoformat()

    db = SessionLocal()
    try:
        record = TrainingJobRecord(
            id=job_id,
            model_id=job_data["model_id"],
            dataset_id=job_data["dataset_id"],
            method=job_data["method"],
            epochs=job_data["epochs"],
            learning_rate=job_data["learning_rate"],
            batch_size=job_data["batch_size"],
            description=job_data.get("description"),
            status="queued",
            progress=0,
            current_stage="queued",
            created_at=now,
        )
        db.add(record)
        db.commit()
        job = job_to_dict(record)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    
    if os.getenv("FORGEML_DRY_RUN", "0") == "1":
        update_job(
            job_id,
            status="dry_run",
            progress=100,
            current_stage="dry_run_complete",
            metrics={
                "dry_run": True,
                "message": "Job creation verified. Training was not started.",
            },
            completed_at=datetime.utcnow().isoformat(),
        )
        return get_training_job(job_id)

    thread = Thread(
        target=execute_training_job,
        args=(job_id,),
        daemon=True,
    )
    thread.start()

    return job


# -------------------------------------------------------------------
# Main training execution pipeline
# -------------------------------------------------------------------

def execute_training_job(job_id: str):

    db = SessionLocal()
    try:
        record = db.get(TrainingJobRecord, job_id)
        if record is None:
            return
        job = job_to_dict(record)
    finally:
        db.close()

    try:

        update_job(job_id, started_at=datetime.utcnow().isoformat())

        # -------------------------------------------------------------
        # Stage 1: Validate
        # -------------------------------------------------------------

        update_job(
            job_id,
            status="running",
            progress=5,
            current_stage="validating",
        )

        db = SessionLocal()
        try:
            model_record = db.get(ModelRecord, job["model_id"])

            if model_record is None:
                raise ValueError(
                    f"Model not found: {job['model_id']}"
                )

            model_name = model_record.model_id
        finally:
            db.close()

        if job["method"].lower() != "lora":
            raise ValueError(
                "V1.1 currently supports only LoRA training."
            )


        # -------------------------------------------------------------
        # Stage 2: Resolve dataset
        # -------------------------------------------------------------

        update_job(
            job_id,
            progress=10,
            current_stage="loading_dataset",
        )

        
        db = SessionLocal()
        try:
            dataset_record = db.get(DatasetRecord, job["dataset_id"])

            if dataset_record is None:
                raise ValueError(
                    f"Dataset not found: {job['dataset_id']}"
                )

            configured_path = dataset_record.file_path
        finally:
            db.close()

        if configured_path:
            dataset_path = Path(configured_path)

            if not dataset_path.is_absolute():
                dataset_path = BACKEND_DIR / dataset_path
        else:
            # Compatibility fallback for existing datasets without a file path.
            dataset_path = DATA_DIR / "local_train.jsonl"

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Training dataset file not found: {dataset_path}"
            )


        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Training dataset not found: {dataset_path}"
            )

        dataset = load_dataset(
            "json",
            data_files=str(dataset_path),
            split="train",
        )

        if len(dataset) == 0:
            raise ValueError("Training dataset is empty.")

        # -------------------------------------------------------------
        # Stage 3: Load tokenizer
        # -------------------------------------------------------------

        update_job(
            job_id,
            progress=20,
            current_stage="loading_tokenizer",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # -------------------------------------------------------------
        # Stage 4: Load model
        # -------------------------------------------------------------

        update_job(
            job_id,
            progress=30,
            current_stage="loading_model",
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        model.config.use_cache = False

        # -------------------------------------------------------------
        # Stage 5: Format + tokenize dataset
        # -------------------------------------------------------------

        update_job(
            job_id,
            progress=40,
            current_stage="preparing_dataset",
        )

        def format_example(example):

            text = (
                "### Instruction:\n"
                f"{example['instruction']}\n\n"
                "### Response:\n"
                f"{example['response']}"
                f"{tokenizer.eos_token}"
            )

            return {"text": text}

        dataset = dataset.map(
            format_example
        )

        def tokenize_example(example):

            tokens = tokenizer(
                example["text"],
                truncation=True,
                max_length=512,
                padding="max_length",
            )

            tokens["labels"] = tokens["input_ids"].copy()

            return tokens

        tokenized_dataset = dataset.map(
            tokenize_example,
            remove_columns=dataset.column_names,
        )

        # -------------------------------------------------------------
        # Stage 6: Configure LoRA
        # -------------------------------------------------------------

        update_job(
            job_id,
            progress=50,
            current_stage="configuring_lora",
        )

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",

            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

        model = get_peft_model(
            model,
            lora_config,
        )

        # Print trainable parameter information.
        model.print_trainable_parameters()

        # -------------------------------------------------------------
        # Stage 7: Configure trainer
        # -------------------------------------------------------------

        update_job(
            job_id,
            progress=55,
            current_stage="preparing_training",
        )

        job_output_dir = (
            OUTPUT_DIR / f"job_{job_id}" / "adapter"
        )

        job_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        training_args = TrainingArguments(
            output_dir=str(job_output_dir),

            num_train_epochs=job["epochs"],

            max_steps=job["max_steps"],

            per_device_train_batch_size=job["batch_size"],

            learning_rate=job["learning_rate"],

            logging_steps=1,

            save_strategy="no",

            report_to=[],

            remove_unused_columns=False,

            fp16=False,

            bf16=False,
        )

        # -------------------------------------------------------------
        # Create real-time ForgeML callback
        # -------------------------------------------------------------

        progress_callback = ForgeMLTrainingCallback(
            job_id
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            callbacks=[
                progress_callback
            ],
        )

        # -------------------------------------------------------------
        # Stage 8: Actual training
        # -------------------------------------------------------------

        update_job(
            job_id,
            progress=60,
            current_stage="training",
        )

        train_result = trainer.train()

        # -------------------------------------------------------------
        # Stage 9: Save LoRA adapter
        # -------------------------------------------------------------

        update_job(
            job_id,
            progress=90,
            current_stage="saving_artifact",
        )

        trainer.save_model(
            str(job_output_dir)
        )

        tokenizer.save_pretrained(
            str(job_output_dir)
        )

        # -------------------------------------------------------------
        # Stage 10: Metrics
        # -------------------------------------------------------------

        metrics = dict(train_result.metrics)

        metrics["training_examples"] = len(dataset)
        metrics["epochs"] = job["epochs"]
        metrics["batch_size"] = job["batch_size"]
        metrics["learning_rate"] = job["learning_rate"]
        metrics["method"] = job["method"]
        metrics["max_steps"] = job["max_steps"]

        # Create the artifact only after metrics have been populated.
        artifact = create_artifact(
            training_job_id=job_id,
            model_id=job["model_id"],
            dataset_id=job["dataset_id"],
            method=job["method"],
            artifact_path=str(job_output_dir),
            metrics=metrics,
        )

        # -------------------------------------------------------------
        # Completed
        # -------------------------------------------------------------

        update_job(
            job_id,

            status="completed",

            artifact_id=artifact["id"],

            progress=100,

            current_stage="completed",

            completed_at=datetime.utcnow().isoformat(),

            metrics=metrics,

            artifact_path=str(job_output_dir),
        )

    except Exception as exc:

        update_job(
            job_id,

            status="failed",

            current_stage="failed",

            error=str(exc),
        )


def get_training_job(job_id: str):
    db = SessionLocal()
    try:
        record = db.get(TrainingJobRecord, job_id)
        return job_to_dict(record) if record else None
    finally:
        db.close()


def get_training_jobs():
    db = SessionLocal()
    try:
        records = (
            db.query(TrainingJobRecord)
            .order_by(TrainingJobRecord.created_at.desc())
            .all()
        )
        return [job_to_dict(record) for record in records]
    finally:
        db.close()


def delete_training_job(job_id: str):
    db = SessionLocal()
    try:
        record = db.get(TrainingJobRecord, job_id)
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