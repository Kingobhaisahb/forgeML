from datetime import datetime
from uuid import uuid4

artifacts = {}


def create_artifact(
    training_job_id: str,
    model_id: str,
    dataset_id: str,
    method: str,
    artifact_path: str,
    metrics: dict | None = None,
):
    artifact_id = str(uuid4())

    artifact = {
        "id": artifact_id,
        "training_job_id": training_job_id,
        "model_id": model_id,
        "dataset_id": dataset_id,
        "method": method,
        "type": "lora_adapter",
        "path": artifact_path,
        "metrics": metrics,
        "created_at": datetime.utcnow().isoformat(),
    }

    artifacts[artifact_id] = artifact

    return artifact


def get_artifact(artifact_id: str):
    return artifacts.get(artifact_id)


def get_artifacts():
    return list(artifacts.values())


def delete_artifact(artifact_id: str):
    if artifact_id not in artifacts:
        return False

    del artifacts[artifact_id]
    return True