import json
import re
import time
from logging import getLogger

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel

from app.common.http_client import async_client
from app.common.mongo import get_db

# LangGraph imports
from app.langgraph_service import get_semantic_graph, run_semantic_chat
from app.utils.storage import (
    add_pqs_file,
    get_answer_match,
    get_question_match,
    read_output,
    store_output,
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


@router.post("/chat/background_semantic")
async def semantic_chat_background(request: SemanticChatRequest,
                                   background_tasks: BackgroundTasks):
    """
    LangGraph-powered semantic chat endpoint.

    Performs semantic search, generates AI response, and returns structured JSON output.
    Workflow: Search → LLM Summarization → JSON Formatting

    Returns:
        JSON object ready for frontend consumption
    """

    tag = time.strftime("%H%M%S", time.localtime())

    background_tasks.add_task(semantic_pipeline, request, tag)
    return {"message":f"Semantic pipeline is running. Use the tag {tag} to retrieve the output." }

def semantic_pipeline(request, tag):

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
                    "error": f"JSON parsing failed: {str(e)}",
                    "raw_output": json_output_string
            }

        else:
            output = {"message":"No JSON output generated"}

    except Exception:
        output = {"message":"Error in semantic chat workflow: {str(e)}"}

    store_output("semantic_chat_" + tag + ".json",output)


@router.get("/chat/semantic_output")
async def semantic_chat_result(tag: str = Query("", description="Semantic Query Tag")):
    """
    Returns the result of the last semantic chat pipeline run if available.
    """
    return read_output("semantic_chat_" + tag + ".json")



@router.post("/chat/semantic")
async def semantic_chat(request: SemanticChatRequest):
    """
    LangGraph-powered semantic chat endpoint.

    Performs semantic search, generates AI response, and returns structured JSON output.
    Workflow: Search → LLM Summarization → JSON Formatting

    Returns:
        JSON object ready for frontend consumption
    """
 #   if df.empty:
 #       raise HTTPException(status_code=500, detail="Data not loaded properly")

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
                "error": f"JSON parsing failed: {str(e)}",
                "raw_output": json_output_string
            }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in semantic chat workflow: {str(e)}"
        ) from e


@router.get("/upload")
async def upload_questions(
    background_tasks: BackgroundTasks,
    pq_file: str = Query(..., description="The name of the file in S3 containing the PQs to insert into the stores")
    ):
    """Add a number of documents to the store using the saved ids"""

    background_tasks.add_task(add_pqs_file, pq_file)
    return {"message":f"Uploading PQs from {pq_file}" }

@router.get("/db")
async def db_query(db=Depends(get_db)):
    await db.example.insert_one({"foo": "bar"})
    data = await db.example.find_one({}, {"_id": 0})
    return {"ok": data}


@router.get("/http")
async def http_query(client=Depends(async_client)):
    resp = await client.get("http://localstack:4566/health")
    return {"ok": resp.status_code}
