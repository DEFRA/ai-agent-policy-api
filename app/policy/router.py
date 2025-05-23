from logging import getLogger
import requests

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query


from app.common.http_client import async_client
from app.common.mongo import get_db

router = APIRouter(prefix="/policy")
logger = getLogger(__name__)


# remove this example route
@router.get("/test")
async def root():
    logger.info("TEST ENDPOINT")
    return {"ok": True}


@router.get("/pq")
async def get_question(
    question_id: int = Query(0, description="Index of PQ")
):
    """Get PQ from written answers api using provided question id"""

    base_url = "https://questions-statements-api.parliament.uk/api"
    endpoint = f"{base_url}/writtenquestions/questions/{question_id}"

    try:
        response = requests.get(
            endpoint,
            headers={"Accept": "application/json",
                     "User-Agent": "Python/Requests"}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching question {question_id}: {e}")
        return {}

@router.get("/db")
async def db_query(db=Depends(get_db)):
    await db.example.insert_one({"foo": "bar"})
    data = await db.example.find_one({}, {"_id": 0})
    return {"ok": data}


@router.get("/http")
async def http_query(client=Depends(async_client)):
    resp = await client.get("http://localstack:4566/health")
    return {"ok": resp.status_code}
