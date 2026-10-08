from fastapi import FastAPI

from app.api.models import router as models_router
from app.api.datasets import router as datasets_router
from app.api.training import router as training_router
from app.api.artifacts import router as artifacts_router
from app.api.evaluations import router as evaluations_router
from app.database import Base, engine
import app.database_models  # Registers all five tables



app = FastAPI(
    title="ForgeML",
    description="AI model training and deployment platform",
    version="0.1.0",
)

Base.metadata.create_all(bind=engine)


app.include_router(models_router)
app.include_router(datasets_router)
app.include_router(training_router)
app.include_router(artifacts_router)
app.include_router(evaluations_router)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ForgeML",
        "version": "0.1.0",
    }