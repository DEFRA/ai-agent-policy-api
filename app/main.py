from contextlib import asynccontextmanager
from logging import getLogger

from fastapi import FastAPI

from app.common.mongo import get_mongo_client

#from app.common.s3 import S3Client
from app.common.tracing import TraceIdMiddleware
from app.config import config
from app.health.router import router as health_router
from app.langgraph_service import build_semantic_chat_graph
from app.policy.router import router as policy_router
from app.utils.storage import check_storage

logger = getLogger(__name__)

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup
    print(f"S3_BUCKET: {config.S3_BUCKET}")

    client = await get_mongo_client()
    logger.info("MongoDB client connected")

    question_store, answer_store = await check_storage()
    # Initialize LangGraph workflow
    semantic_chat_graph = build_semantic_chat_graph()
    print(f"Chat graph {semantic_chat_graph}")
    print("✅ LangGraph semantic chat workflow initialized")

    print("Yielding")
    yield
    # Shutdown
    print("Exiting")
    """
    if s3_client:
        s3_client.close_connection()
        logger.info("S3 client closed")
    """
    if client:
        await client.close()
        logger.info("MongoDB client closed")


app = FastAPI(lifespan=lifespan)

# Setup middleware
app.add_middleware(TraceIdMiddleware)

# Setup Routes
app.include_router(health_router)
app.include_router(policy_router)
