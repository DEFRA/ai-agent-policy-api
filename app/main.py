import os
from contextlib import asynccontextmanager
from logging import getLogger

from fastapi import FastAPI

from app.common.mongo import get_mongo_client
from app.common.tracing import TraceIdMiddleware
from app.config import config as settings
from app.health.router import router as health_router
from app.langgraph_service import build_semantic_chat_graph
from app.policy.router import router as policy_router
from app.utils.storage import check_storage

logger = getLogger(__name__)

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup

    client = await get_mongo_client()
    logger.info("MongoDB client connected")

    if not os.getenv("OPENAI_API_KEY"):
        print("Retrieving OPENAI API key")
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

    question_store, answer_store = await check_storage()
    if question_store is not None:
        # Initialize LangGraph workflow
        semantic_chat_graph = build_semantic_chat_graph()
        if semantic_chat_graph:
            logger.info("✅ LangGraph semantic chat workflow initialized")
    else:
        logger.error("No available Vector Stores")

    yield
    # Shutdown

    if client:
        await client.close()
        logger.info("MongoDB client closed")


app = FastAPI(lifespan=lifespan)

# Setup middleware
app.add_middleware(TraceIdMiddleware)

# Setup Routes
app.include_router(health_router)
app.include_router(policy_router)
