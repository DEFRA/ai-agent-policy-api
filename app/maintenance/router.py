import json
import re
from datetime import datetime, timezone
from logging import getLogger

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel

from app.common.http_client import async_client
from app.common.sync_mongo import get_db, get_item, list_item_ids, replace_item

#from app.common.mongo import get_db
# LangGraph imports
from app.langgraph_service import get_semantic_graph, run_semantic_chat
from app.utils.storage import (
    get_answer_match,
    get_pq_stats,
    get_question_match,
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
        "message": "Policy Maintenance",
        "version": "1.0.0",
    }


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

@router.get("/db_query")
async def get_content(key: str = Query("", description="item key"),
                      collection: str = Query("", description="mongo collection"),
                      name: str = Query("", description="name of data structure")):
    """Retrieves mongo content."""
    item = get_item(tag=key, collection_name=collection, data_name=name)
    return {"Mongo record":item}


@router.get("/db_update")
async def replace_content(item: str = Query("", description="item"),
                      key: str = Query("", description="item key"),
                      collection: str = Query("", description="mongo collection"),
                      name: str = Query("", description="name of data structure")):
    """Retrieves mongo content."""
    item = replace_item(item=item, tag=key, collection_name=collection, data_name=name)
    return {"Mongo record":item}



@router.get("/db_list")
async def list_items(collection: str = Query("", description="mongo collection")):
    """Retrieves mongo content."""
    items = list_item_ids(collection_name=collection)
    return {"Stored Items":items}
