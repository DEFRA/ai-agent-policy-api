import time
from logging import getLogger
from typing import Any

import requests

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
        print(f"Error fetching data: {e}")
        return []


def get_all_question_ids(answering_body_id=None, house="Commons", batch_size=1_000):
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
        break
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
        print(f"Error fetching {question_id=}: {e}")
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
    print(f"Params {answering_body_id}, {house}")
#    all_ids = get_all_question_ids(
#        answering_body_id=answering_body_id, house=house)
    all_ids = [1798613,1798075,1797992,1797598,1798009,1796692,1798097,1796902,1796972,1798010,
               1796975,1798069,1798071,1796977,1797183,1798073,1797614,1796446,1797615,1796447,
               1796349,1797684,1797286,1797862,1797984,1797983,1797982,1797981,1798119,1798158,
               1798160,1797521,1796514,1796217,1795816,1794239,1793717,1791308,1788771,1788834,
               1796687,1797297,1796348,1796363,1796442,1796440]

    # Initialize list to store all question details
    all_questions = []

    error_count = 0

    # Process each ID
    for i, question_id in enumerate(all_ids, 1):
        if error_count > 10:
            print(f"Exceeded error threshold {error_count}")
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
        else:
            error_count += 1

        # Add a small delay to avoid overwhelming the API
        time.sleep(0.1)  # 100ms delay between requests

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
    # Get all question IDs
    print("Fetching PQs")
    print(pq_ids)
    # Initialize list to store all question details
    all_questions = []

    error_count = 0

    # Process each ID
    for i, question_id in enumerate(pq_ids, 1):
        if error_count > 10:
            print(f"Exceeded error threshold {error_count}")
            break
        # Only print progress every 250 questions
        if i % 250 == 0 or i == 1:
            print(
                f"Processing question {i}/{len(pq_ids)} (ID: {question_id})")

        # Get question details
        question_data = get_question_details(question_id)

        if question_data:
            # Extract the 'value' field which contains the actual question data
            if "value" in question_data:
                all_questions.append(question_data["value"])
            else:
                print(
                    f"Warning: No 'value' field found for question {question_id}")
        else:
            error_count += 1

        # Add a small delay to avoid overwhelming the API
        time.sleep(0.1)  # 100ms delay between requests

    return all_questions
