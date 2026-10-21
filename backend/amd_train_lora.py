
import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    data_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    status_path = output_dir / "status.json"

    def write_status(**values):
        payload = {
            "job_id": args.job_id,
            "status": "running",
            **values,
        }
        temp_path = status_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload), encoding="utf-8")
        temp_path.replace(status_path)

    if not torch.cuda.is_available():
        raise RuntimeError("AMD GPU is not available to PyTorch.")

    with data_path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        raise ValueError("Training dataset is empty.")

    for row in records:
        if not isinstance(row.get("instruction"), str) or not isinstance(row.get("response"), str):
            raise ValueError("Every record needs string instruction and response fields.")

    print(f"Dataset: {len(records)} examples", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
    ).to("cuda")
    model.config.use_cache = False

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, config)
    model.train()

    examples = []
    for row in records:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["instruction"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": row["instruction"]},
                {"role": "assistant", "content": row["response"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = tokenizer(
            prompt, add_special_tokens=False, truncation=True, max_length=512
        )["input_ids"]
        full_ids = tokenizer(
            full_text, add_special_tokens=False, truncation=True, max_length=512
        )["input_ids"]

        if len(full_ids) <= len(prompt_ids):
            raise ValueError("An example has no response tokens.")

        labels = full_ids.copy()
        labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

        examples.append((
            torch.tensor(full_ids, dtype=torch.long, device="cuda"),
            torch.tensor(labels, dtype=torch.long, device="cuda"),
        ))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )

    losses = []
    start = time.time()
    total_steps = len(examples) * args.epochs
    step = 0

    write_status(progress=5, stage="training", step=0, total_steps=total_steps)

    for epoch in range(args.epochs):
        epoch_losses = []

        for input_ids, labels in examples:
            optimizer.zero_grad(set_to_none=True)
            result = model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=torch.ones_like(input_ids).unsqueeze(0),
                labels=labels.unsqueeze(0),
            )
            loss = result.loss
            loss.backward()
            optimizer.step()

            step += 1
            loss_value = float(loss.item())
            epoch_losses.append(loss_value)

            progress = 5 + int(85 * step / total_steps)
            write_status(
                progress=progress,
                stage="training",
                step=step,
                total_steps=total_steps,
                epoch=epoch + 1,
                loss=loss_value,
                epoch_average_loss=sum(epoch_losses) / len(epoch_losses),
            )
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Step {step}/{total_steps} | Loss {loss_value:.6f}",
                flush=True,
            )

        losses.append(sum(epoch_losses) / len(epoch_losses))

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "model_id": args.model_id,
        "method": "LoRA",
        "dataset": str(data_path),
        "dataset_examples": len(records),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "epoch_average_losses": losses,
        "elapsed_seconds": round(time.time() - start, 2),
        "gpu": torch.cuda.get_device_name(0),
        "adapter_path": str(output_dir),
    }

    (output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    write_status(progress=100, stage="completed", metrics=metrics, status="completed")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
