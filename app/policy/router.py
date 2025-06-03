import re
from logging import getLogger

import requests
from fastapi import APIRouter, Depends, HTTPException, Query

from app.common.http_client import async_client
from app.common.mongo import get_db
from app.utils.storage import get_answer_match, get_question_match

router = APIRouter(prefix="/policy")
logger = getLogger(__name__)

"""
@router.on_event("startup")
async def startup_event():
    print("STARTUP")
    global question_store
    global answer_store

    try:
        question_store, answer_store = store_documents()
        print("STARTUP: following store creation")

    except Exception as e:
        print(f"Error during startup: {e}")

"""

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

    proxies = {
    "http": "http://localhost:3128",
    "https": "http://localhost:3128",
    }


    base_url = "https://questions-statements-api.parliament.uk/api"
    endpoint = f"{base_url}/writtenquestions/questions/{question_id}"

    try:
        response = requests.get(
            endpoint,
            headers={"Accept": "application/json",
                     "User-Agent": "Python/Requests"},
            timeout=5,
            proxies=proxies
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching question {question_id}: {e}")
        return {}


@router.get("/search/questions")
async def search_questions(
    question: str = Query(..., description="The question to search for"),
    limit: int = Query(5, description="Number of results to return")
):

    """Search for similar questions using cosine similarity"""
    # Regex for case-insensitive match for Affairs with or without trailing comma
    pattern = r"(?i)\baffairs\b[, ]*"

    # Check if the pattern exists first
    if re.search(pattern, question):
        parts = re.split(pattern, question)
        question = parts[1]

    top_results = []
    try:
        # Perform search
        print("Before get question")
        top_results = get_question_match(
                            question=question,
                            limit=limit
                        )
        print(f"After {top_results}")
        return {
            "results": top_results,
            "query": question,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error performing search: {str(e)}") from e


@router.get("/search/answers")
async def search_answers(
    question: str = Query(..., description="The question to search for"),
    limit: int = Query(5, description="Number of results to return")
):
    """Search for similar answers using cosine similarity"""
    top_results = []
    try:
        # Perform search
        top_results = get_answer_match(
                            question=question,
                            limit=limit
                        )

        return {
            "results": top_results,
            "query": question,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error performing search: {str(e)}") from e


@router.get("/db")
async def db_query(db=Depends(get_db)):
    await db.example.insert_one({"foo": "bar"})
    data = await db.example.find_one({}, {"_id": 0})
    return {"ok": data}


@router.get("/http")
async def http_query(client=Depends(async_client)):
    resp = await client.get("http://localstack:4566/health")
    return {"ok": resp.status_code}
