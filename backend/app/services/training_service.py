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

from app.api.models import models
from app.services.artifact_service import create_artifact


# -------------------------------------------------------------------
# In-memory training job registry
# -------------------------------------------------------------------

training_jobs = {}


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


# -------------------------------------------------------------------
# Create training job
# -------------------------------------------------------------------

def create_training_job(job_data: dict):

    job_id = str(uuid4())

    job = {
        "id": job_id,
        "model_id": job_data["model_id"],
        "dataset_id": job_data["dataset_id"],
        "method": job_data["method"],
        "epochs": job_data["epochs"],
        "learning_rate": job_data["learning_rate"],
        "batch_size": job_data["batch_size"],
        "max_steps": FORGEML_MAX_STEPS,
        "description": job_data.get("description"),

        "status": "queued",
        "progress": 0,
        "current_stage": "queued",

        "error": None,

        "created_at": datetime.utcnow().isoformat(),
        "started_at": None,
        "completed_at": None,

        "metrics": None,
        "artifact_path": None,
    }

    training_jobs[job_id] = job

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

    job = training_jobs.get(job_id)

    if job is None:
        return

    try:

        job["started_at"] = datetime.utcnow().isoformat()

        # -------------------------------------------------------------
        # Stage 1: Validate
        # -------------------------------------------------------------

        update_job(
            job_id,
            status="running",
            progress=5,
            current_stage="validating",
        )

        model_record = models.get(job["model_id"])

        if model_record is None:
            raise ValueError(
                f"Model not found: {job['model_id']}"
            )

        if job["method"].lower() != "lora":
            raise ValueError(
                "V1.1 currently supports only LoRA training."
            )

        model_name = model_record["model_id"]

        # -------------------------------------------------------------
        # Stage 2: Resolve dataset
        # -------------------------------------------------------------

        update_job(
            job_id,
            progress=10,
            current_stage="loading_dataset",
        )

        dataset_path = DATA_DIR / "local_train.jsonl"

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

        artifact = create_artifact(
            training_job_id=job_id,
            model_id=job["model_id"],
            dataset_id=job["dataset_id"],
            method=job["method"],
            artifact_path=str(job_output_dir),
            metrics=metrics,
        )

        # -------------------------------------------------------------
        # Stage 10: Metrics
        # -------------------------------------------------------------

        metrics = train_result.metrics

        metrics["training_examples"] = len(dataset)
        metrics["epochs"] = job["epochs"]
        metrics["batch_size"] = job["batch_size"]
        metrics["learning_rate"] = job["learning_rate"]
        metrics["method"] = job["method"]
        metrics["max_steps"] = job["max_steps"]

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


# -------------------------------------------------------------------
# Update job
# -------------------------------------------------------------------

def update_job(job_id: str, **updates):

    if job_id not in training_jobs:
        return

    training_jobs[job_id].update(updates)


# -------------------------------------------------------------------
# Get single job
# -------------------------------------------------------------------

def get_training_job(job_id: str):

    return training_jobs.get(job_id)


# -------------------------------------------------------------------
# Get all jobs
# -------------------------------------------------------------------

def get_training_jobs():

    return list(training_jobs.values())


# -------------------------------------------------------------------
# Delete job
# -------------------------------------------------------------------

def delete_training_job(job_id: str):

    if job_id in training_jobs:

        del training_jobs[job_id]

        return True

    return False
