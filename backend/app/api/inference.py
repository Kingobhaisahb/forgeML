
import threading
import time
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.database import SessionLocal
from app.database_models import ModelRecord
from app.services.artifact_service import get_artifact


router = APIRouter(prefix="/api/inference", tags=["Inference"])

# Cache models to avoid reloading them for every request.
_loaded_models = {}
_model_lock = threading.Lock()

_base_models = {}
_base_model_lock = threading.Lock()


class GenerateRequest(BaseModel):
    artifact_id: str
    prompt: str = Field(min_length=1, max_length=12000)
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class CompareRequest(BaseModel):
    artifact_id: str
    prompt: str = Field(min_length=1, max_length=12000)
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


def get_base_model_id(artifact_id: str) -> str:
    artifact = get_artifact(artifact_id)

    if artifact is None:
        raise ValueError(f"Artifact not found: {artifact_id}")

    db = SessionLocal()
    try:
        model_record = db.get(ModelRecord, artifact["model_id"])

        if model_record is None:
            raise ValueError(
                f"Base model record not found: {artifact['model_id']}"
            )

        base_model_id = model_record.model_id
    finally:
        db.close()

    if not base_model_id:
        raise ValueError("The registered base model has no model_id.")

    return base_model_id


def load_adapter(artifact_id: str):
    artifact = get_artifact(artifact_id)

    if artifact is None:
        raise ValueError(f"Artifact not found: {artifact_id}")

    adapter_path = Path(artifact["path"])

    if not adapter_path.is_absolute():
        adapter_path = Path(__file__).resolve().parents[2] / adapter_path

    if not adapter_path.is_dir():
        raise ValueError(f"Adapter directory not found: {adapter_path}")

    if not (adapter_path / "adapter_config.json").exists():
        raise ValueError("The selected artifact is not a valid PEFT adapter.")

    base_model_id = get_base_model_id(artifact_id)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def load_base_model(artifact_id: str):
    base_model_id = get_base_model_id(artifact_id)

    with _base_model_lock:
        if base_model_id not in _base_models:
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            model = AutoModelForCausalLM.from_pretrained(base_model_id)
            model.eval()

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            _base_models[base_model_id] = (tokenizer, model)

        return _base_models[base_model_id]


def generate_response(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
):
    formatted_prompt = prompt

    if getattr(tokenizer, "chat_template", None):
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0,
    }

    if temperature > 0:
        generation_args["temperature"] = temperature

    started = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(**inputs, **generation_args)

    elapsed = time.perf_counter() - started
    new_tokens = output[0][inputs["input_ids"].shape[1]:]

    return {
        "response": tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip(),
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "generated_tokens": int(new_tokens.shape[0]),
        "generation_time_seconds": round(elapsed, 3),
    }


@router.post("/generate")
def generate(request: GenerateRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=422, detail="Prompt cannot be empty.")

    try:
        with _model_lock:
            if request.artifact_id not in _loaded_models:
                _loaded_models[request.artifact_id] = load_adapter(
                    request.artifact_id
                )

            tokenizer, model = _loaded_models[request.artifact_id]

        result = generate_response(
            tokenizer,
            model,
            request.prompt,
            request.max_new_tokens,
            request.temperature,
        )

        return {
            "artifact_id": request.artifact_id,
            "base_model": str(model.config.name_or_path),
            **result,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(exc).__name__}: {exc}",
        ) from exc


@router.post("/compare")
def compare_models(request: CompareRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=422, detail="Prompt cannot be empty.")

    try:
        # Reuse the original model and the selected adapter.
        base_tokenizer, base_model = load_base_model(request.artifact_id)

        with _model_lock:
            if request.artifact_id not in _loaded_models:
                _loaded_models[request.artifact_id] = load_adapter(
                    request.artifact_id
                )

            adapter_tokenizer, adapter_model = _loaded_models[
                request.artifact_id
            ]

        base_result = generate_response(
            base_tokenizer,
            base_model,
            request.prompt,
            request.max_new_tokens,
            request.temperature,
        )

        fine_tuned_result = generate_response(
            adapter_tokenizer,
            adapter_model,
            request.prompt,
            request.max_new_tokens,
            request.temperature,
        )

        return {
            "artifact_id": request.artifact_id,
            "prompt": request.prompt,
            "base_model": {
                "name": get_base_model_id(request.artifact_id),
                **base_result,
            },
            "fine_tuned_model": {
                "name": get_base_model_id(request.artifact_id) + " + LoRA",
                **fine_tuned_result,
            },
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {type(exc).__name__}: {exc}",
        ) from exc
