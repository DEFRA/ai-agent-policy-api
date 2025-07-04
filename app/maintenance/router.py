import ast
from logging import getLogger

from fastapi import APIRouter, BackgroundTasks, Query

from app.common.sync_mongo import (
    get_item,
    list_item_ids,
    list_timestamp_data,
    replace_item,
)

#from app.common.mongo import get_db
from app.utils.storage import (
    get_pq_stats,
    read_chat_file,
    update_pqs,
)

router = APIRouter(prefix="/maintenance")
logger = getLogger(__name__)


@router.get("/update")
async def update_pqs_in_background(background_tasks: BackgroundTasks):
    """Retrieves PQs based on ids held for re-checking (e.g. for answers to
    previously unanswered questions), and those which are available
    from the Written Questions API but not in the question store.

    Run in the background, as the process can take several minutes.
    """

    background_tasks.add_task(update_pqs)
    return {"message":"Retrieving PQs from Written Questions API." }


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
    """Replaces the item in Mondo identified by the provided key with the Python structure
    that is passed in string form.
    """
    item = replace_item(item=ast.literal_eval(item), tag=key, collection_name=collection, data_name=name)
    return {"Mongo record":item}


@router.get("/db_list")
async def list_items(collection: str = Query("", description="mongo collection")):
    """Lists the keys contained in the named Mongo collection."""
    items = list_item_ids(collection_name=collection)
    return {"Stored Items":items}


@router.get("/get_timestamp_data")
async def get_timestamps():
    """Lists the timestamps of the chat outputs held in the semantic_output collection."""
    return list_timestamp_data()


@router.get("/db_chat_storage")
async def insert_chat(tag: str = Query("", description="chat tag"),
                      timestamp: str = Query("", description="timestamp")):
    """Upserts a stored chat output given the tag (typically, the timestamp).
    Only used for maintenance purposes.
    """
    data = read_chat_file(tag)
    to_store = {"chat":data, "timestamp":timestamp}
    item = replace_item(item=to_store, tag=tag)
    return {"Mongo record key":item}
