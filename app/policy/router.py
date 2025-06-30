import json
import re
from datetime import datetime, timezone
from logging import getLogger

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel

from app.common.http_client import async_client
from app.common.mongo import get_db

# LangGraph imports
from app.langgraph_service import get_semantic_graph, run_semantic_chat
from app.utils.storage import (
    get_answer_match,
    get_pq_stats,
    get_question_match,
    load_status,
    read_output,
    store_output,
    update_pqs,
)

router = APIRouter(prefix="/policy")
logger = getLogger(__name__)

# Pydantic models for request/response

class SemanticChatRequest(BaseModel):
    question: str


@router.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "message": "Semantic Search API",
        "version": "1.0.0",
        "endpoints": [
            "/search/questions",
            "/search/answers",
            "/chat/semantic",
            "/health"
        ]
    }


@router.get("/search/questions")
async def search_questions(
    question: str = Query(..., description="The question to search for"),
    limit: int = Query(5, description="Number of results to return")
):
    """Search for similar questions using the cosine similarity measure in FAISS"""
    # Regex for case-insensitive match for Affairs with or without trailing comma
    pattern = r"(?i)\baffairs\b[, ]*"

    # Check if the pattern exists first
    if re.search(pattern, question):
        parts = re.split(pattern, question)
        question = parts[1]

    top_results = []
    try:
        # Perform search
        top_results = get_question_match(
                            question=question,
                            limit=limit
                        )
        return {
            "results": top_results,
            "query": question,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error performing search: {e}") from e


@router.get("/search/answers")
async def search_answers(
    question: str = Query(..., description="The question to search for"),
    limit: int = Query(5, description="Number of results to return")
):
    """Search for similar answers using the cosine similarity measure in FAISS.
    This is a useful technique to supplement the question search, as answers to
    some questions may provide an insight.
    """
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
            status_code=500, detail=f"Error performing search: {e}") from e


@router.post("/chat/background_semantic")
async def semantic_chat_background(request: SemanticChatRequest,
                                   background_tasks: BackgroundTasks) -> dict[str, str]:
    """
    Triggers a background task to:
      perform a semantic search
      generate an AI response
      return structured JSON output.

    Returns:
        A dictionary containing the time-based tag for use in querying the
        generated result.
    """
    tag =  datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    background_tasks.add_task(semantic_pipeline, request, tag)
    return {"message":f"{tag}" }


def semantic_pipeline(request: SemanticChatRequest, tag: str):
    """
    LangGraph-powered semantic chat endpoint.

    Performs semantic search, generates AI response, and returns structured JSON output.
    Workflow: Search → LLM Summarization → JSON Formatting

    The pipeline creates output in JSON format that is stored in S3 for
    collection by other services.

    Args:
        request: the Pydantic model containing the question
        tag: string identifier to identify the output stored by this function
    """

    semantic_chat_graph = get_semantic_graph()

    if semantic_chat_graph is None:
        raise HTTPException(
                status_code=500, detail="LangGraph workflow not initialized")
    try:

        # Execute LangGraph workflow
        result = run_semantic_chat(semantic_chat_graph, request.question)

        # Get the JSON output from the LLM
        json_output_string = result.get("json_output", "")

        if json_output_string:
            # Parse the JSON string into an actual JSON object
            try:
                # Clean the JSON string if it has markdown formatting
                cleaned_json = json_output_string.strip()
                if cleaned_json.startswith("```json"):
                    cleaned_json = cleaned_json[7:]  # Remove ```json
                if cleaned_json.endswith("```"):
                    cleaned_json = cleaned_json[:-3]  # Remove ```
                cleaned_json = cleaned_json.strip()

                # Parse into actual JSON object and return

                output = json.loads(cleaned_json)

            except json.JSONDecodeError as e:
                # If JSON parsing fails, return a structured fallback
                output = {
                    "query": request.question,
                    "answer": result.get("response", "Error generating response"),
                    "search_results": result.get("search_results", []),
                    "error": f"JSON parsing failed: {e}",
                    "raw_output": json_output_string
            }

        else:
            output = {"message":"No semantic chat output generated"}

    except Exception:
        output = {"message":"Error in semantic chat workflow: {e}"}

    store_output(tag, output)


@router.get("/chat/semantic_output")
def semantic_chat_result(tag: str = Query("", description="Semantic Query Tag")):
    """
    Returns the result of the last semantic chat pipeline run if available.
    """
    return read_output(tag)



@router.post("/chat/semantic")
async def semantic_chat(request: SemanticChatRequest):
    """
    Direct invocation of the semantic pipeline, but may time out.

    The semantic_chat_background & semantic_chat_result functions are
    a replacement for this function, by invoking the pipeline as a
    background process.

    LangGraph-powered semantic chat endpoint.

    Performs semantic search, generates AI response, and returns structured JSON output.
    Workflow: Search → LLM Summarization → JSON Formatting

    Returns:
        JSON object ready for frontend consumption
    """
    semantic_chat_graph = get_semantic_graph()

    if semantic_chat_graph is None:
        raise HTTPException(
            status_code=500, detail="LangGraph workflow not initialized")

    try:
        # Execute LangGraph workflow
        result = run_semantic_chat(semantic_chat_graph, request.question)

        # Get the JSON output from the LLM
        json_output_string = result.get("json_output", "")

        if not json_output_string:
            raise HTTPException(
                status_code=500, detail="No JSON output generated")

        # Parse the JSON string into an actual JSON object
        try:
            # Clean the JSON string if it has markdown formatting
            cleaned_json = json_output_string.strip()
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:]  # Remove ```json
            if cleaned_json.endswith("```"):
                cleaned_json = cleaned_json[:-3]  # Remove ```
            cleaned_json = cleaned_json.strip()

            # Parse into actual JSON object and return

            return json.loads(cleaned_json)

        except json.JSONDecodeError as e:
            # If JSON parsing fails, return a structured fallback
            return {
                "query": request.question,
                "answer": result.get("response", "Error generating response"),
                "search_results": result.get("search_results", []),
                "error": f"JSON parsing failed: {e}",
                "raw_output": json_output_string
            }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in semantic chat workflow: {e}"
        ) from e

@router.get("/update")
async def answer_status(background_tasks: BackgroundTasks):
    """Retrieves PQs from the status file."""

    background_tasks.add_task(update_pqs)
    return {"message":"Retrieving PQs from status file" }

@router.get("/stats")
async def show_stats():
    """Retrieves count of stored PQs and ids to be checked."""
    stats = await get_pq_stats()
    return {"PQ stats":stats}

@router.get("/store_status")
async def store_status():
    """Retrieves ids from status file and inserts into mongo."""
    result = await load_status()
    return {"load_status":result}

@router.get("/db")
async def db_query(db=Depends(get_db)):
    await db.example.insert_one({"foo": "bar"})
    data = await db.example.find_one({}, {"_id": 0})
    return {"ok": data}


@router.get("/http")
async def http_query(client=Depends(async_client)):
    resp = await client.get("http://localstack:4566/health")
    return {"ok": resp.status_code}
