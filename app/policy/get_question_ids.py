import requests


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


if __name__ == "__main__":
    # Example usage with different parameters
    # Basic usage
    # ids = get_question_ids(answering_body_id=13)
    # print(f"Found {len(ids)} question IDs (basic):")
    # print(ids)

    # # Usage with custom parameters
    # ids_custom = get_question_ids(
    #     answering_body_id=13,
    #     skip=0,
    #     take=100,
    #     house="Lords"
    # )
    # print(f"\nFound {len(ids_custom)} question IDs (with custom params):")
    # print(ids_custom)

    # Get all IDs (will handle pagination automatically)
    print("\nFetching all IDs (this might take a while)...")
    all_ids = get_all_question_ids(answering_body_id=13)
    print(f"Total IDs found: {len(all_ids)}")
