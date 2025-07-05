import json
from logging import getLogger
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter
from langchain.vectorstores import FAISS
from langchain.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.common.s3 import S3Client
from app.common.sync_mongo import add_item, delete_item, get_item

from .policy_retrieval import (
    get_all_question_ids,
    get_specific_question_details,
)

# this is a simple hack to avoid a ruff complaint about temp file usage
TMP = "tmp"
QUESTION_STORE_DIR = "/" + TMP + "/question_store_4/"
ANSWER_STORE_DIR =  "/" + TMP + "/answer_store_4/"

question_store = None
answer_store = None
s3_client = S3Client()
embed_model = None

router = APIRouter(prefix="/policy")
logger = getLogger(__name__)


def populate_embeddable_questions(df: pd.DataFrame) -> pd.DataFrame:
    """Split the question on variants of "Affairs", and set the
    embeddable_question column to the remaining part of the question.
    """
    df_split = df["questionText"].str.split(r"(?i)\baffairs\b[, ]*", n=1, expand=True)

    df.loc[df["questionText"].str.contains(r"(?i)\baffairs\b[, ]*", regex=True), "embeddable_question"] = df_split[1].fillna("")
    df.loc[~df["questionText"].str.contains(r"(?i)\baffairs\b[, ]*", regex=True, na=False), "embeddable_question"] = df_split[0]

    return df


def create_documents(df: pd.DataFrame) -> tuple(list[Any]):
    """Creates Langgraph Documents from the PQs.
    Each Document contains a page_content text string,
    an Id, and a collection of metadata.

    As we want both the question and the answer of a PQ to be searchable,
    two Documents are created for each PQ, one for the question and one
    for the answer.

    Each PQ has an index, which we use as the Document Id.

    For the question Document, we associate a set of metadata, but for
    the associated answer, only the page_content and Id are populated.

    Note: the answer text is also included as an item of metadata in
    the associated question Document to simplify later processing.

    Returns:
        list[Document]: the Documents containing the question text
        list[Document]: the Documents containing the answer text
        list[int]: the ids of PQs successfully captured in Documents
        list[int]: the ids of PQs failing to be captured in Documents
    """
    question_documents = []
    answer_documents = []

    success_ids = []
    failed_ids = []

    for _index, question in df.iterrows():
        try:
            question_documents.append(
                Document(
                id=question["id"],
                page_content=question["embeddable_question"],
                metadata={"answer": question["answerText"],
                          "asking_member_id": question["askingMemberId"],
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
            success_ids.append(id)

        except Exception as e:
            logger.error("Error creating Document for question %s: %s", question, e)
            failed_ids.append(id)

    return question_documents, answer_documents, success_ids, failed_ids


def get_missing_pq_ids() -> list[int]:
    """Returns PQ ids retrieved from the Written Questions API that
    don't exist in the Questions vector store.
    """
    stored_ids = get_stored_pq_ids()

    try:
        all_ids = get_all_question_ids(answering_body_id=13, house="Commons")
    except Exception as e:
        logger.error("Error retrieving ids: %s", e)
        return []

    all_set = set(all_ids)
    stored_set = set(stored_ids)

    # Worth checking if the store now has ids not being returned from Gov API
    logger.info("Extra ids: \n %s", stored_set - all_set)
    return list(all_set - stored_set)

def get_stored_pq_ids() -> list[int]:
    """Retrieves the PQ ids from the Question store - we should have the
    same ids in the Answer store.
    """

#    store_dir = "/" + TMP + QUESTION_STORE_DIR
    store_dir = QUESTION_STORE_DIR
    exists = s3_client.check_object_existence(store_dir + "index.faiss")

    pids = []

    if not exists:
        logger.error("No PQs are stored in %s", store_dir)
        return pids

    try:
        vector_store = load_store(store_dir)

        pids = [int(pid) for pid in vector_store.index_to_docstore_id.values()]

    except Exception as e:
        logger.error("Failed to load vector store at %s: %s", store_dir, e)

    return pids

def remove_pq_vectors(store, documents: list[Document]) -> tuple[list[int], list[int]]:
    """Deletes the vectors in the passed vector store whose ids
    are contained in the supplied Documents.
    Returns a success status.
    """
    del_ids = [int(d.id) for d in documents]
    success_ids = []
    failure_ids = []
    # To avoid one failed deletion ending in failure
    # delete single PQs rather than as a batch
    logger.info("Del ids %s",del_ids[:5])
    for pid in del_ids:
        try:
            store.delete([pid])
            success_ids.append(pid)
        except Exception as e:
            logger.error("Deletion of vector failed: %s", e)
            failure_ids.append(pid)
    return success_ids, failure_ids



async def update_pqs():
    """Updates any PQs which are answered but were previously unanswered.
    Then retrieves and inserts any PQs not currently in the vector stores.
    """
    logger.info("*** update_pqs begun ***")
    await update_answers()
    await insert_new_pqs()
    logger.info("*** update_pqs ended ***")


async def update_answers():
    """Reads the collection of PQ ids marked for checking. This is currently only used
    to check for PQs that have no answer, but may have been answered since they were
    inserted in the vector stores.

    The PQs associated with these ids are retrieved, with any failed retrievals recorded
    by having their ids recorded for subsequent attempts.
    """

    ids = await read_status_data()

    if len(ids) > 0:
        ids = list(set(ids))
        logger.info("The following PQs will be checked for answers: \n%s", ids)
    else:
        logger.info("No statuses to check")
        return

    questions, not_retrieved_ids = get_specific_question_details(ids)

    if not_retrieved_ids:
        logger.error("The following PQs requiring answers were not retrieved successfully: %s", not_retrieved_ids)

    # now update the PQs that do have answers while storing the ids that still don't
    await update_stores(questions, not_retrieved_ids)


async def insert_new_pqs():
    """Uses the list of PQ ids not currently in the question store to retrieve
    the missing questions, and to update the vector stores with those questions.
    """
    missing_ids = get_missing_pq_ids()
    logger.info("Count of missing PQs: %s", len(missing_ids))
    logger.info("Missing ids %s",missing_ids)
    if not missing_ids:
        return

    questions, not_retrieved_ids = get_specific_question_details(missing_ids)

    # Any PQs not successfully retrieved should be picked up on the next run
    if not_retrieved_ids:
        logger.error("The following PQs were not retrieved successfully: %s", not_retrieved_ids)

    await update_stores(questions)


def process_pqs(questions: list[dict]) -> list[int]:
    """Any question not having an answer has its id recorded for later checking.
    The answer field is populated with an empty html paragraph, as the answer_store
    embedding requires a non-null string, and the general format of answers uses <p></p>
    delimiters.

    In order to simplify the text processing prior to embedding, we use a pandas
    DataFrame.

    LangChain Documents are created from the questions and inserted into the
    Question and Answer vector stores.

    The list of unanswered PQ ids is returned for persistence and subsequent checking.
    """
    global question_store, answer_store

    pq_count = len(questions)
    logger.info("In process_pqs with %s questions",pq_count)
    if pq_count == 0:
        return []

    revisit_ids = []

#    question_dir = "/" + TMP + QUESTION_STORE_DIR
#    answer_dir = "/" + TMP + ANSWER_STORE_DIR

    for question in questions:
        if not question["answerText"]:
            revisit_ids.append(question["id"])
            # add an empty paragraph as a null answertext cannot be indexed
            question["answerText"] = "<p></p>"
        else:
            logger.info("PQ %s has an answer",question["id"])

    try:
        # The necessary PQ transformations are simpler using pandas

        df = pd.DataFrame(questions)
        df = populate_embeddable_questions(df)

        question_documents, answer_documents, success_ids, failed_ids = create_documents(df)
        logger.info("Failed ids %s",failed_ids[:5])

        # if some PQs have failed the document creation, they're stored for later
        revisit_ids.extend(failed_ids)

#        question_store, question_ids_not_added = update_vector_store(question_documents, question_dir)
#        answer_store, answer_ids_not_added = update_vector_store(answer_documents, answer_dir)
        question_store, question_ids_not_added = update_vector_store(question_documents, QUESTION_STORE_DIR)
        answer_store, answer_ids_not_added = update_vector_store(answer_documents, ANSWER_STORE_DIR)
    except Exception as e:
        logger.error("Failed to update stores with PQs \n%s:\n %s",success_ids,e)

    # assemble list of ids not added at some stage

    return list(set(revisit_ids).union(set(question_ids_not_added),set(answer_ids_not_added)))


async def get_pq_stats():
    """Returns the count of stored PQs and the number of ids to be checked."""
    to_check_count = len(await read_status_data())
    stored_count = len(get_stored_pq_ids())

    return {"stored_pq_count": stored_count,
            "further_check_count": to_check_count}


async def update_stores(questions, to_check_ids=None):
    """Inserts the questions into the vector stores.
    Updates the list of PQ ids to be checked and writes the list to Mongo
    """
    if not to_check_ids:
        to_check_ids = []
    if questions:
        not_inserted_ids = process_pqs(questions)
        logger.info("To check ids %s",to_check_ids)
        logger.info("Not inserted ids %s",not_inserted_ids)
        to_check_ids.extend(not_inserted_ids)


    delete_item("to_check", "maintenance")
    logger.info("The following PQs will be stored in mongo: \n%s", to_check_ids)
    add_item(to_check_ids, "to_check", "maintenance")


async def read_status_data() -> list[str]:
    """Read the saved ids from the status item in mongo."""
    ids = []

    try:
        ids = get_item("to_check", collection_name="maintenance", data_name="data")
        ids = [int(pid) for pid in ids]
        ids = set(ids)
        logger.info("Ids to check %s",ids)
    except Exception as e:
        logger.error("Failed to manage the mongo status item: %s",e)
    return ids


def read_chat_file(tag: str) -> dict:
    """Downloads the named file from S3, and returns the
    contents as a list of strings.
    The file is deleted to avoid accidental reuse.
    """
    # load file containing the ids to check using pq api
    # hack to overcome ruff insistence on avoiding /tmp
#    store_dir = "/" + TMP + QUESTION_STORE_DIR

    filename = "semantic_chat_" + tag + ".json"

    target =  QUESTION_STORE_DIR + filename

    exists = s3_client.check_object_existence(target)

    if not exists:
       return {"message":"Semantic Chat output not yet available, please try again soon."}

    create_directory_if_necessary(QUESTION_STORE_DIR)

    s3_client.download_file(target, target)

    with open(target) as file:
        result = json.load(file)
        logger.info("File generated output: %s", result)

    return result


def create_vector_store(documents: list[Document],
                        store_dir: str) -> FAISS:
    """
    Create a FAISS vector store from the supplied Documents using
    the specified embedding model. The measure of inter-vector
    distance is Cosine Similarity.

    The vector store is saved in the local file system and then
    uploaded to S3.
    """

    vector_store = FAISS.from_documents(
                                      documents,
                                      embedding=embed_model,
                                      distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
                                      )
    # Save vector store
    vector_store.save_local(store_dir)

    s3_client.upload_file(store_dir + "index.faiss")
    s3_client.upload_file(store_dir + "index.pkl")
    return vector_store


def update_vector_store(documents: list[Document],
                        store_dir: str) -> tuple[FAISS, list[int]]:
    """
    Add Documents to an existing FAISS vector store.
    The store is saved locally and then the index files
    are uploaded to S3.

    If the store dosn't exist, create it.

    If the update fails, the ids of the PQs to be inserted are returned to
    the caller, so that a retry using those PQs can be provisioned.
    """
    logger.info("Updating store with %s PQs", len(documents))

    # save these ids in case of update failure
    ids_to_be_inserted = [int(doc.id) for doc in documents]
    added_ids = []

    vector_store = None
    exists = s3_client.check_object_existence(store_dir + "index.faiss")

    if not exists:
        logger.info("Creating new FAISS store at %s", store_dir)

        vector_store = create_vector_store(documents, store_dir)
        logger.info("Created %s", vector_store)

        return vector_store, ids_to_be_inserted

    try:
        vector_store = load_store(store_dir)
        num_documents = len(vector_store.index_to_docstore_id)
        logger.info("Total number of documents prior to update: %s", num_documents)

        # remove any that have already been inserted, as upsert is not available
        # Note: where the PQ has not been inserted previously, the deletion will fail. This is unimportant.
        success_ids, failure_ids = remove_pq_vectors(vector_store, documents)
        logger.info("These ids were successfully removed %s\nThese were not removed %s", success_ids, failure_ids)

        added_ids = vector_store.add_documents(documents=documents)
        # Save vector store
        vector_store.save_local(store_dir)

        num_documents = len(vector_store.index_to_docstore_id)
        logger.info("Total number of documents post update: %s", num_documents)

        s3_client.upload_file(store_dir + "index.faiss")
        s3_client.upload_file(store_dir + "index.pkl")
    except Exception as e:
        logger.error("Failed to update vector store at %s: %s", store_dir, e)

    added_ints = [int(pid) for pid in added_ids]
    ids_not_added = list(set(ids_to_be_inserted) - set(added_ints))
    if ids_not_added:
        logger.info("The following ids remain to be added %s", ids_not_added)
    else:
        logger.info("All PQs were added successfully")

    return vector_store, ids_not_added



def load_store(store_dir: str) -> FAISS:
    """
    Instantiates a FAISS vector store from its constituent index files,
    which are downloaded from S3.
    """
    store_path = Path(store_dir)
    if not store_path.exists():
        store_path.mkdir()

    faiss_file = store_dir + "index.faiss"
    pickle_file = store_dir + "index.pkl"

    s3_client.download_file(faiss_file, faiss_file)
    s3_client.download_file(pickle_file, pickle_file)

    try:
       return FAISS.load_local(store_dir, embed_model,allow_dangerous_deserialization=True)
    except Exception as e:
       logger.error("Error creating FAISS: %s", e)
       return None


def create_directory_if_necessary(directory_name: str):
    """Creates local directory. """
    try:
        path = Path(directory_name)
        if not path.exists():
            path.mkdir()

    except Exception as e:
        logger.error("Error creating %s directory: %s", path, e)


async def check_storage():
    """Load the Question and Answer vector stores from S3.
    Also, initialise the associated global variables.
    """

    global question_store, answer_store, embed_model


    # hack to overcome ruff insistence on avoiding /tmp

#    question_dir = "/" + TMP + QUESTION_STORE_DIR
#    answer_dir = "/" + TMP + ANSWER_STORE_DIR

    embed_model = OpenAIEmbeddings(
                       model="text-embedding-3-small",
                 )
    question_index = QUESTION_STORE_DIR + "index.faiss"

    exists = s3_client.check_object_existence(question_index)

    if not exists:
        logger.error("Vector stores not located")
    else:
        question_store = load_store(QUESTION_STORE_DIR)
        answer_store = load_store(ANSWER_STORE_DIR)

    return question_store, answer_store


def get_question_match(question, limit) -> list[dict]:
    """Given a question, find the specified number of matches in the Question store."""
    documents =  question_store.similarity_search_with_score(
                            query=question,
                            k=limit
                        )

    return [{"id":doc.id,
              "question":doc.page_content,
              "answer":doc.metadata.get("answer",""),
              "date_tabled":doc.metadata.get("date_tabled",""),
              "uin":doc.metadata.get("uin",""),
              "score":float(score)} for doc,score in documents]


def get_answer_match(question, limit):
    documents = answer_store.similarity_search_with_score(
                            query=question,
                            k=limit
                        )
    return [{"id":doc.id, "answer":doc.page_content, "score":float(score)} for doc,score in documents]


def store_output(tag: str, json_content: dict):
    """Write the output from the agentic workflow to Mongo """
    to_store = {"chat":json_content, "timestamp":tag}

    add_item(item=to_store, tag=tag)


def read_output(tag: str):
    """Locate the chat output in Mongo using the provided identifier.
    If it's not there, look for the chat output as an S3 file.
    """
    result = None
    # First try mongoDB
    try:
        stored_result = get_item(tag)
        if stored_result.get("timestamp","") is not None:
            result = stored_result.get("chat",{})
        else:
            result = stored_result
        logger.info("Mongo result %s",result)
    except Exception as e:
        logger.error("Failed to manage the mongo chat for %s: %s", tag, e)

    # Some early outputs were written to S3 - but should exist in Mongo - so shouldn't ever need this
    if not result:
        result = read_chat_file(tag)

    return result
