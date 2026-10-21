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

import json

import shlex

import subprocess

import time

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

        "compute_target": (

        (record.metrics or {}).get("compute_target", "local")

    ),

    }

def update_job(job_id: str, **updates):

    db = SessionLocal()

    try:

        record = db.get(TrainingJobRecord, job_id)

        if record is None:

            return

        # Preserve the selected compute target when training metrics update.

        if "metrics" in updates:

            existing_metrics = dict(record.metrics or {})

            incoming_metrics = updates["metrics"] or {}

            updates["metrics"] = {

                **existing_metrics,

                **incoming_metrics,

            }

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

            metrics={

                "compute_target": job_data.get("compute_target", "local")

            },

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

    compute_target = job_data.get("compute_target", "local")

    if compute_target == "amd_mi300x":

        worker = execute_amd_training_job

    elif compute_target == "local":

        worker = execute_training_job

    else:

        raise ValueError(f"Unsupported compute target: {compute_target}")

    thread = Thread(

        target=worker,

        args=(job_id,),

        daemon=True,

    )

    thread.start()

    return get_training_job(job_id)

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

def execute_amd_training_job(job_id: str):

    """Run a ForgeML LoRA job on the existing remote AMD GPU worker."""

    host = os.getenv("FORGEML_AMD_HOST", "165.245.135.243")

    key_path = Path(

        os.getenv(

            "FORGEML_AMD_SSH_KEY",

            str(Path.home() / ".ssh" / "forgeml_amd"),

        )

    )

    container = os.getenv("FORGEML_AMD_CONTAINER", "forgeml-trainer")

    remote_root = f"/root/forgeml-data/jobs/{job_id}"

    remote_dataset = f"/workspace/forgeml/data/jobs/{job_id}.jsonl"

    remote_output = f"/workspace/forgeml/artifacts/jobs/{job_id}"

    remote_log = f"{remote_output}/training.log"

    remote_exit = f"{remote_output}/exitcode"

    remote_status = f"{remote_output}/status.json"

    local_job_dir = OUTPUT_DIR / f"amd_job_{job_id}"

    local_job_dir.mkdir(parents=True, exist_ok=True)

    def run_ssh(command, timeout=60):
        command_preview = command[:180].replace("\\n", " ")
        print(
            f"[ForgeML AMD SSH] host={host} timeout={timeout}s "
            f"command={command_preview}",
            flush=True,
        )

        try:
            result = subprocess.run(
                [
                    "ssh", "-i", str(key_path),
                    "-o", "BatchMode=yes",
                    "-o", "ConnectTimeout=10",
                    "-o", "ServerAliveInterval=15",
                    "-o", "ServerAliveCountMax=2",
                    f"root@{host}", command,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            print(
                f"[ForgeML AMD SSH TIMEOUT] host={host} "
                f"timeout={timeout}s command={command_preview}",
                flush=True,
            )
            raise RuntimeError(
                f"SSH subprocess timed out after {timeout}s "
                f"for command: {command_preview}"
            ) from exc

        if result.returncode != 0:
            error = (
                result.stderr.strip()
                or result.stdout.strip()
                or f"SSH command failed: {command_preview}"
            )
            print(
                f"[ForgeML AMD SSH FAILED] exit={result.returncode} "
                f"error={error[:500]}",
                flush=True,
            )
            raise RuntimeError(error)

        return result.stdout.strip()

    def run_scp(source, destination, recursive=False, timeout=300):
        command = [
            "scp", "-i", str(key_path),
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10",
        ]
        if recursive:
            command.append("-r")
        command.extend([str(source), str(destination)])

        print(
            f"[ForgeML AMD SCP] source={source} "
            f"destination={destination} timeout={timeout}s",
            flush=True,
        )

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            print(
                f"[ForgeML AMD SCP TIMEOUT] destination={destination} "
                f"timeout={timeout}s",
                flush=True,
            )
            raise RuntimeError(
                f"SCP subprocess timed out after {timeout}s "
                f"to destination: {destination}"
            ) from exc

        if result.returncode != 0:
            error = (
                result.stderr.strip()
                or result.stdout.strip()
                or "SCP transfer failed"
            )
            print(
                f"[ForgeML AMD SCP FAILED] exit={result.returncode} "
                f"error={error[:500]}",
                flush=True,
            )
            raise RuntimeError(error)

    try:

        update_job(

            job_id,

            status="running",

            progress=2,

            current_stage="connecting_to_amd",

            started_at=datetime.utcnow().isoformat(),

        )

        db = SessionLocal()

        try:

            record = db.get(TrainingJobRecord, job_id)

            if record is None:

                raise ValueError(f"Training job not found: {job_id}")

            job = job_to_dict(record)

            model = db.get(ModelRecord, job["model_id"])

            dataset = db.get(DatasetRecord, job["dataset_id"])

            if model is None:

                raise ValueError(f"Model not found: {job['model_id']}")

            if dataset is None:

                raise ValueError(f"Dataset not found: {job['dataset_id']}")

            model_name = model.model_id

            dataset_file = dataset.file_path

        finally:

            db.close()

        if job["method"].lower() != "lora":

            raise ValueError("AMD worker currently supports LoRA only.")

        # Resolve the local dataset file.

        if dataset_file:

            dataset_path = Path(dataset_file)

            if not dataset_path.is_absolute():

                dataset_path = BACKEND_DIR / dataset_path

        else:

            dataset_path = DATA_DIR / "local_train.jsonl"

        if not dataset_path.is_file():

            raise FileNotFoundError(

                f"Training dataset not found: {dataset_path}"

            )

        # Validate the model ID before placing it in a remote shell command.

        import re

        if not re.fullmatch(

            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",

            model_name,

        ):

            raise ValueError(f"Unsupported Hugging Face model ID: {model_name}")

        update_job(

            job_id,

            progress=8,

            current_stage="uploading_dataset",

        )

        run_ssh(

            f"mkdir -p {shlex.quote(remote_root)} "

            f"&& docker exec {shlex.quote(container)} mkdir -p "

            f"{shlex.quote(remote_output)} "

            f"/workspace/forgeml/data/jobs"

        )

        run_scp(

            dataset_path,

            f"root@{host}:{remote_root}/dataset.jsonl",

        )

        run_ssh(

            f"docker cp {shlex.quote(remote_root + '/dataset.jsonl')} "

            f"{shlex.quote(container + ':' + remote_dataset)}"

        )

        # Reset previous status and exit markers for this job.

        run_ssh(

            f"docker exec {shlex.quote(container)} "

            f"/bin/bash -lc "

            f"{shlex.quote('rm -f ' + shlex.quote(remote_status) + ' '

                           + shlex.quote(remote_exit) + ' '

                           + shlex.quote(remote_log))}"

        )

        update_job(

            job_id,

            progress=15,

            current_stage="starting_amd_training",

        )

        # Use a detached process so the API can poll progress while training runs.

        training_command = " ".join([

            "/usr/bin/python3",

            shlex.quote("/workspace/forgeml/amd_train_lora.py"),

            "--model-id", shlex.quote(model_name),

            "--dataset", shlex.quote(remote_dataset),

            "--output-dir", shlex.quote(remote_output),

            "--epochs", str(int(job["epochs"])),

            "--learning-rate", str(float(job["learning_rate"])),

            "--job-id", shlex.quote(job_id),

        ])

        inner_command = (

            f"{training_command} > {shlex.quote(remote_log)} 2>&1; "

            f"echo $? > {shlex.quote(remote_exit)}"

        )

        run_ssh(

            f"docker exec -d {shlex.quote(container)} "

            f"/bin/bash -lc {shlex.quote(inner_command)}"

        )

        # Poll the worker's atomic status file.

        deadline = time.monotonic() + 3600

        final_status = None

        while time.monotonic() < deadline:

            try:

                raw_status = run_ssh(

                    f"docker exec {shlex.quote(container)} "

                    f"cat {shlex.quote(remote_status)}"

                )

                final_status = json.loads(raw_status)

                live_metrics = {

                    "compute_target": "amd_mi300x",

                    "step": final_status.get("step"),

                    "total_steps": final_status.get("total_steps"),

                    "epoch": final_status.get("epoch"),

                    "loss": final_status.get("loss"),

                    "epoch_average_loss": final_status.get(

                        "epoch_average_loss"

                    ),

                }

                update_job(

                    job_id,

                    progress=int(final_status.get("progress", 15)),

                    current_stage=final_status.get("stage", "training"),

                    metrics=live_metrics,

                )

                if final_status.get("status") == "completed":

                    break

            except (RuntimeError, json.JSONDecodeError):

                # The status file may not exist during model loading.

                pass

            try:

                exit_code = run_ssh(

                    f"docker exec {shlex.quote(container)} "

                    f"cat {shlex.quote(remote_exit)}"

                )

                if exit_code.strip() not in ("", "0"):

                    log = run_ssh(

                        f"docker exec {shlex.quote(container)} "

                        f"cat {shlex.quote(remote_log)}"

                    )

                    raise RuntimeError(

                        f"AMD training exited with code {exit_code}:\n{log}"

                    )

            except RuntimeError as exc:

                if "No such file" not in str(exc):

                    raise

            time.sleep(2)

        else:

            raise TimeoutError("AMD training exceeded the 60-minute limit.")

        if not final_status or final_status.get("status") != "completed":

            raise RuntimeError("AMD worker did not report training completion.")

        update_job(

            job_id,

            progress=94,

            current_stage="downloading_adapter",

        )

        # Copy adapter out of the container, then back to this PC.

        run_ssh(

            f"docker cp {shlex.quote(container + ':' + remote_output)} "

            f"{shlex.quote(remote_root + '/output')}"

        )

        adapter_path = local_job_dir / "output"

        run_scp(

            f"root@{host}:{remote_root}/output",

            local_job_dir,

            recursive=True,

            timeout=600,

        )

        if not (adapter_path / "adapter_model.safetensors").is_file():

            raise FileNotFoundError(

                f"Adapter file was not received at {adapter_path}"

            )

        metrics = dict(final_status.get("metrics") or {})

        metrics["compute_target"] = "amd_mi300x"

        artifact = create_artifact(

            training_job_id=job_id,

            model_id=job["model_id"],

            dataset_id=job["dataset_id"],

            method="lora",

            artifact_path=str(adapter_path),

            metrics=metrics,

        )

        update_job(

            job_id,

            status="completed",

            progress=100,

            current_stage="completed",

            completed_at=datetime.utcnow().isoformat(),

            metrics=metrics,

            artifact_path=str(adapter_path),

            artifact_id=artifact["id"],

        )

    except Exception as exc:

        update_job(

            job_id,

            status="failed",

            current_stage="failed",

            error=str(exc),

            completed_at=datetime.utcnow().isoformat(),

        )

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