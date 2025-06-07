from contextlib import asynccontextmanager
from logging import getLogger

from fastapi import FastAPI

from app.common.mongo import get_mongo_client

#from app.common.s3 import S3Client
from app.common.tracing import TraceIdMiddleware
from app.config import config
from app.health.router import router as health_router
from app.policy.router import router as policy_router
from app.utils.storage import check_storage, get_pq_ids

logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup
    print(f"S3_BUCKET: {config.S3_BUCKET}")
    """
    s3_client = S3Client()
    s3_ok = s3_client.check_connection()
    """
    client = await get_mongo_client()
    logger.info("MongoDB client connected")

    question_store, answer_store = await check_storage()

    pq_ids = await get_pq_ids()
    print(f"Retrieved {len(pq_ids)} PQ ids.")


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
