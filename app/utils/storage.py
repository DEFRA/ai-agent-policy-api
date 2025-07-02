import csv
import json
import os
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

QUESTION_STORE_DIR="/question_store_4/"
ANSWER_STORE_DIR="/answer_store_4/"

TMP = "tmp"

IDS_FILE = "pq_ids_5.csv"
STATUS_FILE = "pq_status.csv"
UPDATE_FILE = "pq_update.csv"

question_store = None
answer_store = None

pq_ids = []

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
    stored_ids = get_stored_pq_ids()
    logger.info("The stored PQs count: %s", len(stored_ids))

    try:
        all_ids = get_all_question_ids(answering_body_id=13, house="Commons")
        logger.info("Retrieved %s PQs from parliament api", len(all_ids))
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
    same ids in the Answer store
    """
    s3_client = S3Client()
    embed_model = OpenAIEmbeddings(
                       model="text-embedding-3-small",
                    )

    store_dir = "/" + TMP + QUESTION_STORE_DIR
    exists = s3_client.check_object_existence(store_dir + "index.faiss")

    pids = []

    if not exists:
        logger.error("No PQs are stored in %s", store_dir)
        return pids

    try:
        vector_store = load_store(s3_client, store_dir, embed_model)

        pids = [int(pid) for pid in vector_store.index_to_docstore_id.values()]

    except Exception as e:
        logger.error("Failed to load vector store at %s: %s", store_dir, e)

    return pids

def remove_pq_vectors(store, documents: list[Document]) -> tuple[list[int], list[int]]:
    """Deletes the vectors in the passed vector store whose ids
    are contained in the supplied Documents.
    Returns a success status.
    """
    del_ids = [d.id for d in documents]
    success_ids = []
    failure_ids = []
    # To avoid one failed deletion ending in failure
    # delete single PQs rather than as a batch
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
    logger.info("Starting update_pqs ========================")
    await update_answers()
    logger.info("Pre insert_new_pqs ========================")
    await insert_new_pqs()
    logger.info("Ending update_pqs ========================")


async def insert_new_pqs():
    """Uses the list of PQ ids not currently in the question store to retrieve
    the missing questions, and to update the vector stores with those questions.
    """
    missing_ids = get_missing_pq_ids()
    logger.info("Count of missing PQs: %s", len(missing_ids))
    if not missing_ids:
        return

    questions, not_retrieved_ids = get_specific_question_details(missing_ids)

    # Any PQs not successfully retrieved should be picked up on the next run
    if not_retrieved_ids:
        logger.error("The following PQs were not retrieved successfully: %s", not_retrieved_ids)

    await update_stores(questions)


async def update_answers():
    """Reads the collection of PQ ids marked for checking. This is currently only used
    to check for PQs that have no answer, but may have been answered since they were
    inserted in the vector stores.

    The PQs associated with these ids are retrieved, with any failed retrievals recorded
    by having their ids recorded for subsequent attempts.
    """
#    ids = read_status_file(STATUS_FILE, delete=True)
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

    s3_client = S3Client()
    embed_model = OpenAIEmbeddings(
                       model="text-embedding-3-small",
                    )

    question_dir = "/" + TMP + QUESTION_STORE_DIR
    answer_dir = "/" + TMP + ANSWER_STORE_DIR

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

        # if some PQs have failed the document creation, they're stored for later
        revisit_ids.extend(failed_ids)

        question_store, question_ids_not_added = update_vector_store(s3_client, question_documents, embed_model, question_dir)
        answer_store, answer_ids_not_added = update_vector_store(s3_client, answer_documents, embed_model, answer_dir)
    except Exception as e:
        logger.error("Failed to update stores with PQs \n%s:\n %s",success_ids,e)

    # assemble list of ids not added at some stage

    return list(set(revisit_ids).union(set(question_ids_not_added),set(answer_ids_not_added)))


async def get_pq_stats():
#    to_check_count = len(read_status_file(STATUS_FILE, delete=False))
    to_check_count = len(await read_status_data())
    logger.info("To check count %s", to_check_count)

    stored_count = len(get_stored_pq_ids())
    return {"stored_pq_count": stored_count,
            "further_check_count": to_check_count}

async def update_stores(questions, to_check_ids=None):
    """Inserts the questions into the vector stores.
    Updates the list of PQ ids to be checked and writes the list
    to a file in S3.
    """
    if not to_check_ids:
        to_check_ids = []
    if questions:
        not_inserted_ids = process_pqs(questions)
        to_check_ids.extend(not_inserted_ids)

#    write_ids_file(STATUS_FILE, to_check_ids)
    delete_item("to_check", "maintenance")
    logger.info("The following PQs will be stored in mongo: \n%s", to_check_ids)
#    db_status = {"data":to_check_ids}
    add_item(to_check_ids, "to_check", "maintenance")


async def read_status_data() -> list[str]:
    """Read the saved ids from the status item in mongo."""
    ids = []

    try:
        """
        status_item = await get_item("to_check", "maintenance")
        logger.info("Status item %s",status_item)
        content = status_item.get("content",{})
        ids = content.get("data",[])
        """
        ids = get_item("to_check", collection_name="maintenance", data_name="data")
        ids = [int(pid) for pid in ids]
        ids = set(ids)
        logger.info("Ids to check %s",ids)
    except Exception as e:
        logger.error("Failed to manage the mongo status item: %s",e)
    return ids

def read_status_file(filename: str, delete: bool = False) -> list[str]:
    """Downloads the named file from S3, and returns the
    contents as a list of strings.
    The file is deleted to avoid accidental reuse.
    """
    # load file containing the ids to check using pq api
    # hack to overcome ruff insistence on avoiding /tmp
    store_dir = "/" + TMP + QUESTION_STORE_DIR
    file = store_dir + filename

    lines = []

    s3_client = S3Client()

    exists = s3_client.check_object_existence(file)

    if not exists:
        logger.error("File %s not found - exiting!", file)
    else:
        create_directory_if_necessary(store_dir)

        try:
            s3_client.download_file(file, file)

            with open(file,encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    lines.append(int(row[0]))
            logger.info("Read %s items from file.", len(lines) )
        except Exception as e:
            logger.error("Error downloading/reading %s from S3: %s", file, e)

        if delete:
            try:
                # now remove the file
                os.remove(file)
            except Exception as e:
                logger.error("Error deleting %s: %s", file, e)

    return lines


def read_chat_file(tag: str) -> dict:
    """Downloads the named file from S3, and returns the
    contents as a list of strings.
    The file is deleted to avoid accidental reuse.
    """
    # load file containing the ids to check using pq api
    # hack to overcome ruff insistence on avoiding /tmp
    store_dir = "/" + TMP + QUESTION_STORE_DIR
    filename = "semantic_chat_" + tag + ".json"
    s3_client = S3Client()

    target =  store_dir + filename

    exists = s3_client.check_object_existence(target)

    if not exists:
       return {"message":"Semantic Chat output not yet available, please try again soon."}

    create_directory_if_necessary(store_dir)

    s3_client.download_file(target, target)

    with open(target) as file:
        result = json.load(file)
        logger.info("File generated output: %s", result)

    return result


def write_ids_file(filename:str, ids:list[str]):
    store_dir = "/" + TMP + QUESTION_STORE_DIR
    id_file = store_dir + filename

    try:
        with open(id_file, "w") as csvfile:
            writer = csv.writer(csvfile)
            for pid in ids:
                writer.writerow([pid])
    except Exception as e:
        logger.error("Error storing ids in file %s: %s", id_file, e)
        return

    s3_client = S3Client()

    try:
        s3_client.upload_file(id_file)
    except Exception as e:
        logger.error("Error storing %s in S3: %s", id_file, e)


def create_vector_store(s3_client: S3Client,
                        documents: list[Document],
                        embed_model: OpenAIEmbeddings,
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


def update_vector_store(s3_client: S3Client,
                        documents: list[Document],
                        embed_model: OpenAIEmbeddings,
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
    ids_to_be_inserted = [doc.id for doc in documents]
    added_ids = []

    vector_store = None
    exists = s3_client.check_object_existence(store_dir + "index.faiss")

    if not exists:
        logger.info("Creating new FAISS store at %s", store_dir)

        vector_store = create_vector_store(s3_client, documents, embed_model, store_dir)
        logger.info("Created %s", vector_store)

        return vector_store, ids_to_be_inserted

    try:
        vector_store = load_store(s3_client, store_dir, embed_model)
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


def load_store(s3_client: S3Client,
               store_dir: str,
               embed_model: OpenAIEmbeddings) -> FAISS:
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
    try:
        path = Path(directory_name)
        if not path.exists():
            path.mkdir()

    except Exception as e:
        logger.error("Error creating %s directory: %s", path, e)


async def add_pqs_file(filename: str):
    """
    Downloads the specified file of PQs from S3, and then
    update the question and answer vector stores from the Documents.

    Finally, upload the index files to S3.
    """
    global question_store, answer_store

    # quick hack to overcome the ruff dislike of explicitly using /tmp
    temp_dir = "/" + TMP + "/"
    target = temp_dir + filename
    question_dir = "/" + TMP + QUESTION_STORE_DIR
    answer_dir = "/" + TMP + ANSWER_STORE_DIR

    s3_client = S3Client()
    exists = s3_client.check_object_existence(target)

    if not exists:
        logger.error("Source file %s does not exist.", target)
        return

    create_directory_if_necessary(temp_dir)

    s3_client.download_file(target, target)

    # The necessary PQ transformations are simpler using pandas
    df = pd.read_csv(target)
    df = populate_embeddable_questions(df)

    embed_model = OpenAIEmbeddings(
                       model="text-embedding-3-small",
                    )

    try:
        question_documents, answer_documents, success_ids, failed_ids = create_documents(df)
        question_store, question_ids_not_added = update_vector_store(s3_client, question_documents, embed_model, question_dir)
        answer_store, answer_ids_not_added = update_vector_store(s3_client, answer_documents, embed_model, answer_dir)
    except Exception as e:
        logger.error("Failed to update stores with data from %s : %s", target, e)

    return

async def get_pq_ids():
    global pq_ids

    # hack to overcome ruff insistence on avoiding /tmp
    store_dir = "/" + TMP + QUESTION_STORE_DIR
    pq_ids_file = store_dir + IDS_FILE

    s3_client = S3Client()

    exists = s3_client.check_object_existence(pq_ids_file)

    if not exists:
        logger.info("Retrieving ids")
        try:
            pq_ids = get_all_question_ids(answering_body_id=13, house="Commons")
            logger.info("Retrieved %s PQs from parliament api", len(pq_ids))
        except Exception as e:
            logger.error("Error retrieving ids: %s", e)

        create_directory_if_necessary(store_dir)

        try:
            logger.info("Writing csv file of ids %s", pq_ids_file)
            with open(pq_ids_file, "w") as csvfile:
                writer = csv.writer(csvfile)
                for pid in pq_ids:
                    writer.writerow([pid])
        except Exception as e:
            logger.error("Error storing ids in file %s: %s", pq_ids_file, e)
        try:
            s3_client.upload_file(pq_ids_file)
        except Exception as e:
            logger.error("Error storing %s in S3: %s", pq_ids_file, e)

    else:
        create_directory_if_necessary(store_dir)

        try:
            pq_ids = []
            s3_client.download_file(pq_ids_file, pq_ids_file)

            with open(pq_ids_file) as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    pq_ids.append(int(row[0]))
            logger.info("Read %s PQ ids from file %s.", len(pq_ids), pq_ids_file)
        except Exception as e:
            logger.error("Error downloading/reading %s from S3: %s", pq_ids_file, e)

    return pq_ids


async def check_storage():

    global question_store, answer_store

    s3_client = S3Client()

    # hack to overcome ruff insistence on avoiding /tmp

    question_dir = "/" + TMP + QUESTION_STORE_DIR
    answer_dir = "/" + TMP + ANSWER_STORE_DIR

    embed_model = OpenAIEmbeddings(
                       model="text-embedding-3-small",
                 )
    question_index = question_dir + "index.faiss"

    exists = s3_client.check_object_existence(question_index)

    if not exists:
        logger.error("Vector stores not located")
    else:
        question_store = load_store(s3_client, question_dir, embed_model)
        answer_store = load_store(s3_client, answer_dir, embed_model)

    return question_store, answer_store


def get_question_match(question, limit):
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

def store_output(tag, json_content):
    filename = "semantic_chat_" + tag + ".json"
    s3_client = S3Client()
    # hack to overcome ruff insistence on avoiding /tmp

    store_dir = "/" + TMP + QUESTION_STORE_DIR
    create_directory_if_necessary(store_dir)

    target =  store_dir + filename

    with open(target, "w") as f:
        json.dump(json_content, f)

    s3_client.upload_file(target)
    # now double up with mongo
    add_item(item=json_content, tag=tag)

def read_output(tag):
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

    if not result:
        result = read_chat_file(tag)

    return result

async def load_status():
    ids = read_status_file(STATUS_FILE, delete=False)

    if len(ids) > 0:
        logger.info("The following PQs will be stored: \n%s", ids)
#        db_status = {"data":ids}
        add_item(ids, "to_check", "maintenance")
    else:
        logger.info("No statuses to check")
        return
