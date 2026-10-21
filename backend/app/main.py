
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.models import router as models_router
from app.api.datasets import router as datasets_router
from app.api.training import router as training_router
from app.api.artifacts import router as artifacts_router
from app.api.evaluations import router as evaluations_router
from app.api.inference import router as inference_router

from app.routers.history import router as history_router
from app.routers.experiments import router as experiments_router

from app.database import Base, engine
import app.database_models


app = FastAPI(
    title="ForgeML",
    description="AI model training and deployment platform",
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Create any missing database tables, including experiments.
Base.metadata.create_all(bind=engine)


# Register API routers.
app.include_router(models_router)
app.include_router(datasets_router)
app.include_router(training_router)
app.include_router(artifacts_router)
app.include_router(evaluations_router)
app.include_router(inference_router)
app.include_router(history_router)
app.include_router(experiments_router)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ForgeML",
        "version": "0.1.0",
    }
