from logging import getLogger
from typing import Optional

from fastapi import Depends
from pymongo import MongoClient
from pymongo.database import Database

from app.common.tls import custom_ca_certs
from app.config import config

logger = getLogger(__name__)

client: Optional[MongoClient] = None
db: Optional[Database] = None


def get_mongo_client() -> MongoClient:
    """Get a Synchronous Client - the Asynchronous alternative
    doesn't work well when dependent functions are run as
    background jobs by FastAPI.
    """
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
            client = MongoClient(config.mongo_uri, tlsCAFile=cert)
        else:
            logger.info("Creating MongoDB client")
            client = MongoClient(config.mongo_uri)

        logger.info("Testing MongoDB connection to %s", config.mongo_uri)
        check_connection(client)
    return client


def get_db(client: MongoClient = Depends(get_mongo_client)) -> Database:
    global db
    if db is None:
        db = client.get_database(config.mongo_database)
        logger.info("Database: %s",db)
    return db


def check_connection(client: MongoClient):
    database = get_db(client)
    response = database.command("ping")
    logger.info("MongoDB PING %s", response)


def add_item(item: dict, tag: str, collection_name: str = "semantic_output", data_name: str = "data") -> str:
    """Add a json-style object to a collection with the specified data name.
    As the typical item is a chat output, the collection defaults to semantic_output.
    """
    collection = db[collection_name]
    item_dict = {data_name:item}
    to_store = {"_id": tag, "content": item_dict}

    stored_item = (collection.insert_one(to_store))
    stored_id = stored_item.inserted_id
    logger.info("Stored item %s", stored_id)
    return stored_id


def get_item(tag: str, collection_name: str = "semantic_output", data_name: str = "data") -> dict:
    """Retrieve an object from the specified collection using the tag as key."""
    collection = db[collection_name]
    result = {}
    try:
        item = collection.find_one({"_id": tag})

        if item is not None:
            content = item.get("content",{})
            result = content.get(data_name,[])
    except Exception as e:
        logger.error("Error in get_item with tag %s and collection %s: %s", tag, collection_name, e)
    return result

def delete_item(tag: str, collection_name: str = "semantic_output") -> dict:
    collection = db[collection_name]

    return collection.delete_one({"_id": tag})

def replace_item(item, tag: str, collection_name: str = "semantic_output", data_name: str = "data"):
    delete_item(tag, collection_name)
    return add_item(item, tag, collection_name, data_name)

def list_item_ids(collection_name: str = "semantic_output"):
    """Mongo items have ids specified by the _id key."""
    collection = db[collection_name]
    return collection.distinct("_id")

def list_timestamp_data(collection_name: str = "semantic_output"):
    """Return the timestamp data from the objects in the collection.
    Default use case is to retrieve the timestamps of the chat outputs.
    """
    collection = db[collection_name]
    cursor = collection.find({})
    timestamps = []
    for item in cursor:
        content = item.get("content",{})
        result = content.get("data",[])
        timestamp = result.get("timestamp","")
        timestamps.append(timestamp)
    return {"timestamps":sorted(timestamps)}

