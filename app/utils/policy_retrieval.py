import time
from logging import getLogger
from typing import Any

import requests

logger = getLogger(__name__)


def get_question_ids(answering_body_id=None, skip=0, take=10_000, house="Commons", tabled_from=None) -> list[int]:
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

    if tabled_from is not None:
        params["tabledWhenFrom"] = tabled_from

    try:
        response = requests.get(
            endpoint,
            headers={"Accept": "application/json",
                     "User-Agent": "Python/Requests"},
            params=params,
            timeout=500,
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
        logger.error("Error fetching data: %s", e)
        return []


def get_all_question_ids(answering_body_id=None, house="Commons", batch_size=5_000, tabled_from=None) -> list[int]:
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
        batch_ids = get_question_ids(
            answering_body_id=answering_body_id,
            skip=skip,
            take=batch_size,
            house=house,
            tabled_from=tabled_from
        )

        if not batch_ids:  # If we get no results, we've reached the end
            break

        all_ids.extend(batch_ids)

        if len(batch_ids) < batch_size:  # If we got fewer results than requested, we've reached the end
            break

        skip += batch_size

    logger.info("Total IDs retrieved: %s", len(all_ids))
    return all_ids


def get_question_details(question_id: str) -> dict[str, Any]:
    """
    Fetch detailed information for a specific question ID
    Args:
        question_id (str): The ID of the question to fetch
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

    logger.info("Getting question id %s", question_id)

    try:
        response = requests.get(
            endpoint,
            headers={"Accept": "application/json",
                     "User-Agent": "Python/Requests"},
            timeout=5,
            proxies=proxies
        )
        response.raise_for_status()
        logger.info("Retrieved %s",response)
        return response.json()
    except Exception as e:
        logger.error("Error fetching %s: %s", question_id, e)
        return {}


def get_all_question_details(answering_body_id: int = None, house: str = "Commons", tabled_from: str=None) -> list[dict]:
    """
    Fetch detailed information for all questions and save to a DataFrame
    Args:
        answering_body_id (int, optional): Specific answering body ID to filter by
        house (str, optional): House to query ('Commons' or 'Lords')
        tabled_from (str, optional): the minimum tabled date for PQs
    Returns:
        pd.DataFrame: DataFrame containing all question details
    """
    # Get all question IDs
    logger.info("Fetching all questions with the following constraints:")
    logger.info("Params answering_body_id: %s, \nhouse: %s, tabled_from: %s", answering_body_id, house, tabled_from)
    all_ids = get_all_question_ids(
        answering_body_id=answering_body_id, house=house, tabled_from=tabled_from)

    # Initialize list to store all question details
    all_questions = []

    error_count = 0

    # Process each ID
    for _i, question_id in enumerate(all_ids, 1):

        # Get question details
        question_data = get_question_details(question_id)

        if question_data:
            # Extract the 'value' field which contains the actual question data
            if "value" in question_data:
                all_questions.append(question_data["value"])
            else:
                logger.warning("Warning: No 'value' field found for question %s", question_id)
        else:
            error_count += 1

        # Add a small delay to avoid overwhelming the API
        time.sleep(0.2)  # 200ms delay between requests

    return all_questions


def get_specific_question_details(pq_ids):
    """
    Fetch detailed information for all questions and save to a DataFrame
    Args:
        answering_body_id (int, optional): Specific answering body ID to filter by
        house (str, optional): House to query ('Commons' or 'Lords')
    Returns:
        pd.DataFrame: DataFrame containing all question details
    """
    # Initialize list to store all question details
    all_questions = []

    failed_ids = []

    # Process each ID
    for _i, question_id in enumerate(pq_ids, 1):
        # Get question details
        question_data = get_question_details(question_id)

        if question_data:
            # Extract the 'value' field which contains the actual question data
            if "value" in question_data:
                all_questions.append(question_data["value"])
            else:
                logger.warning("Warning: No 'value' field found for question %s", question_id)
        else:
            failed_ids.append(question_id)

        # Add a small delay to avoid overwhelming the API
        time.sleep(0.2)  # 200ms delay between requests

    logger.info("Failed ids: %s", failed_ids)

    return all_questions, failed_ids
