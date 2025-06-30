from logging import getLogger
from typing import Optional

from fastapi import Depends
from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from app.common.tls import custom_ca_certs
from app.config import config

logger = getLogger(__name__)

client: Optional[AsyncMongoClient] = None
db: Optional[AsyncDatabase] = None


async def get_mongo_client() -> AsyncMongoClient:
    global client
    if client is None:
        # Use the custom CA Certs from env vars if set.
        # We can remove this once we migrate to mongo Atlas.
        cert = custom_ca_certs.get(config.mongo_truststore)
        if cert:
            logger.info(
                "Creating MongoDB client with custom TLS cert %s",
                config.mongo_truststore,
            )
            client = AsyncMongoClient(config.mongo_uri, tlsCAFile=cert)
        else:
            logger.info("Creating MongoDB client")
            client = AsyncMongoClient(config.mongo_uri)

        logger.info("Testing MongoDB connection to %s", config.mongo_uri)
        await check_connection(client)
    return client


async def get_db(client: AsyncMongoClient = Depends(get_mongo_client)) -> AsyncDatabase:
    global db
    if db is None:
        db = client.get_database(config.mongo_database)
    return db


async def check_connection(client: AsyncMongoClient):
    database = await get_db(client)
    response = await database.command("ping")
    logger.info("MongoDB PING %s", response)


async def add_item(item: dict, tag: str, collection_name: str = "semantic_output", data_name: str = "data"):
    collection = db[collection_name]
    item_dict = {data_name:item}
    to_store = {"_id": tag, "content": item_dict}

    stored_item = (await collection.insert_one(to_store))
    stored_id = stored_item.inserted_id
    logger.info("Stored item %s", stored_id)


async def get_item(tag: str, collection_name: str = "semantic_output", data_name: str = "data") -> dict:
    collection = db[collection_name]
    logger.info("Found collection %s",collection)
    result = {}
    try:
        item = await collection.find_one({"_id": tag})
        logger.info("Retrieved item %s",item)
        if item is not None:
            content = item.get("content",{})
            result = content.get(data_name,[])
    except Exception as e:
        logger.error("Error in get_item with tag %s and collection %s: %s", tag, collection_name, e)
    return result

async def delete_item(tag: str, collection_name: str = "semantic_output") -> dict:
    collection = db[collection_name]

    return await collection.delete_one({"_id": tag})
