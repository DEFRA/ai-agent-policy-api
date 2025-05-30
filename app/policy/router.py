import re
import time
from logging import getLogger
from typing import Any

import pandas as pd
import requests
from fastapi import APIRouter, Depends, HTTPException, Query
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from app.common.http_client import async_client
from app.common.mongo import get_db

question_store = None
answer_store = None

router = APIRouter(prefix="/policy")
logger = getLogger(__name__)


def get_question_ids(answering_body_id=None, skip=0, take=10_000, house="Commons"):
    """
    Fetch written question IDs from the Parliament API
    Args:
        answering_body_id (int, optional): Specific answering body ID to filter by
        skip (int, optional): Number of records to skip (for pagination)
        take (int, optional): Number of records to return
        house (str, optional): House to query ('Commons' or 'Lords')
    Returns:
        list: List of question IDs
    """
    base_url = "https://questions-statements-api.parliament.uk/api"
    endpoint = f"{base_url}/writtenquestions/questions"

    proxies = {
        "http": "http://localhost:3128",
        "https": "http://localhost:3128",
    }

    params = {
        "house": house,
        "take": take,
        "skip": skip
    }

    if answering_body_id is not None:
        params["answeringBodies"] = str(answering_body_id)

    try:
        print(f"Fetching ids with skip {skip}")
        response = requests.get(
            endpoint,
            headers={"Accept": "application/json",
                     "User-Agent": "Python/Requests"},
            params=params,
            timeout=5,
            proxies=proxies
        )
        response.raise_for_status()

        data = response.json()
        questions = data.get("results", [])

        # Extract just the IDs

        return [
            question["value"]["id"]
            for question in questions
            if isinstance(question, dict) and "value" in question and "id" in question["value"]
        ]

    except Exception as e:
        print(f"Error fetching data: {e}")
        return []


def get_all_question_ids(answering_body_id=None, house="Commons", batch_size=10_000):
    """
    Fetch all written question IDs from the Parliament API, handling pagination
    Args:
        answering_body_id (int, optional): Specific answering body ID to filter by
        house (str, optional): House to query ('Commons' or 'Lords')
        batch_size (int, optional): Number of records to fetch per request
    Returns:
        list: List of all question IDs
    """
    all_ids = []
    skip = 0

    while True:
        print(f"Fetching batch of {batch_size} records, skipping {skip}...")
        batch_ids = get_question_ids(
            answering_body_id=answering_body_id,
            skip=skip,
            take=batch_size,
            house=house
        )

        if not batch_ids:  # If we get no results, we've reached the end
            break

        all_ids.extend(batch_ids)
        print(f"Retrieved {len(batch_ids)} IDs in this batch")

        if len(batch_ids) < batch_size:  # If we got fewer results than requested, we've reached the end
            break

        skip += batch_size

    print(f"Total IDs retrieved: {len(all_ids)}")
    return all_ids


def get_question_details(question_id: int) -> dict[str, Any]:
    """
    Fetch detailed information for a specific question ID
    Args:
        question_id (int): The ID of the question to fetch
    Returns:
        dict: The question details
    """
#    http_proxy = settings.HTTPS_PROXY

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


def get_all_question_details(answering_body_id: int = None, house: str = "Commons"):
    """
    Fetch detailed information for all questions and save to a DataFrame
    Args:
        answering_body_id (int, optional): Specific answering body ID to filter by
        house (str, optional): House to query ('Commons' or 'Lords')
    Returns:
        pd.DataFrame: DataFrame containing all question details
    """
    # Get all question IDs
    print("Fetching all question IDs...")
    all_ids = get_all_question_ids(
        answering_body_id=answering_body_id, house=house)

    # Initialize list to store all question details
    all_questions = []

    # Process each ID
    for i, question_id in enumerate(all_ids, 1):

        if i > 1000:
            break
        # Only print progress every 250 questions
        if i % 250 == 0 or i == 1:
            print(
                f"Processing question {i}/{len(all_ids)} (ID: {question_id})")

        # Get question details
        question_data = get_question_details(question_id)

        if question_data:
            # Extract the 'value' field which contains the actual question data
            if "value" in question_data:
                all_questions.append(question_data["value"])
            else:
                print(
                    f"Warning: No 'value' field found for question {question_id}")

        # Add a small delay to avoid overwhelming the API
        time.sleep(0.1)  # 100ms delay between requests

    # use Pandas for text manipulation
    df = pd.DataFrame(all_questions)
    df = populate_embeddable_questions(df)
    df = populate_embeddable_answers(df)

    return create_documents(df)


def populate_embeddable_questions(df):
    df_split = df["questionText"].str.split(r"(?i)\baffairs\b[, ]*", n=1, expand=True)

    df.loc[df["questionText"].str.contains(r"(?i)\baffairs\b[, ]*", regex=True), "embeddable_question"] = df_split[1].fillna("")
    df.loc[~df["questionText"].str.contains(r"(?i)\baffairs\b[, ]*", regex=True, na=False), "embeddable_question"] = df_split[0]

    return df

def populate_embeddable_answers(df):

    df["answerText"] = df["answerText"].replace(to_replace="<p>", value=" ")
    df["answerText"] = df["answerText"].replace(to_replace="</p>", value=" ")
    return df


def create_documents(df):
    question_documents = []
    answer_documents = []

    for _index, question in df.iterrows():

        try:
            question_documents.append(
                Document(
                id=question["id"],
                page_content=question["embeddable_question"],
                metadata={"asking_member_id": question["askingMemberId"],
                          "date_tabled": question["dateTabled"],
                          "date_for_answer": question["dateForAnswer"],
                          "uin": question["uin"],
                          "is_named_day": question["isNamedDay"],
                          "answering_member_id": question["answeringMemberId"],
                          "date_answered": question["dateAnswered"],
                          "heading": question["heading"]
                         }
                )
            )
            answer_documents.append(
                Document(
                id=question.get("id"),
                page_content=question["answerText"],
                )
            )
        except Exception as e:
            print(f"Error fetching question {question}: {e}")

    return question_documents, answer_documents


def create_vector_store(documents):
    embeddings = OpenAIEmbeddings(
                       model="text-embedding-3-small",
                 )
    return InMemoryVectorStore.from_documents(
                                      documents,
                                      embedding=embeddings,
                                      )


def store_documents(answering_body_id=13):
    question_documents, answer_documents = get_all_question_details(answering_body_id)
    question_store = create_vector_store(question_documents)
    answer_store = create_vector_store(answer_documents)
    return question_store, answer_store


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
        top_results = question_store.similarity_search_with_score(
                            query=question,
                            k=limit
                        )

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
        top_results = answer_store.similarity_search_with_score(
                            query=question,
                            k=limit
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
