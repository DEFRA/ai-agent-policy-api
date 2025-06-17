import csv
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

from .policy_retrieval import get_all_question_ids, get_specific_question_details

QUESTION_STORE_DIR="/question_store_4/"
ANSWER_STORE_DIR="/answer_store_4/"

TMP = "tmp"

IDS_FILE = "pq_ids_5.csv"
STATUS_FILE = "pq_status.csv"

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
#        print(f"Creating embedding for {question['id']}")
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
            print(f"Error creating Document for question {question}: {e}")
            failed_ids.append(id)

    return question_documents, answer_documents, success_ids, failed_ids


async def get_pq_status():
    status_ids = get_ids_from_file(STATUS_FILE)

    if len(status_ids) == 0:
        print("No statuses to check")

    questions, not_retrieved_ids = get_specific_question_details(status_ids)
    for question in questions:
        print(question["answerText"])


def get_ids_from_file(filename):
    # load file containing the ids to check using pq api
    # hack to overcome ruff insistence on avoiding /tmp
    store_dir = "/" + TMP + QUESTION_STORE_DIR
    id_file = store_dir + filename

    status_ids = []

    s3_client = S3Client()

    exists = s3_client.check_object_existence(id_file)

    if not exists:
        print(f"PQ id file {id_file} not found - exiting!")
    else:
        create_directory_if_necessary(store_dir)

        try:
            s3_client.download_file(id_file, id_file)

            with open(id_file) as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    status_ids.append(int(row[0]))
            print(f"Read {len(status_ids)} PQ ids from file.")
        except Exception as e:
            print(f"Error downloading/reading {id_file} from S3: {e}")

    return status_ids


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
                        store_dir: str) -> FAISS:
    """
    Add Documents to an existing FAISS vector store.
    The store is saved locally and then the index files
    are uploaded to S3.

    If the store dosn't exist, create it.
    """
    vector_store = None
    exists = s3_client.check_object_existence(store_dir + "index.faiss")

    if not exists:
        print(f"Creating new FAISS store at {store_dir}")

        vector_store = create_vector_store(s3_client, documents, embed_model, store_dir)
        print(f"Created {vector_store}")

        return vector_store

    try:
        vector_store = load_store(s3_client, store_dir, embed_model)

        vector_store.add_documents(documents=documents)
        # Save vector store
        vector_store.save_local(store_dir)

        num_documents = len(vector_store.index_to_docstore_id)
        print(f"Total number of documents: {num_documents}")

        s3_client.upload_file(store_dir + "index.faiss")
        s3_client.upload_file(store_dir + "index.pkl")
    except Exception as e:
        print(f"Failed to update vector store at {store_dir}: {e}")

    return vector_store


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
       print(f"Error creating FAISS: {e}")
       return None


def create_directory_if_necessary(directory_name: str):
    try:
        path = Path(directory_name)
        if not path.exists():
            path.mkdir()

    except Exception as e:
        print(f"Error creating {path} directory: {e}")


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
        print(f"Source file {target} does not exist.")
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
        question_store = update_vector_store(s3_client, question_documents, embed_model, question_dir)
        answer_store = update_vector_store(s3_client, answer_documents, embed_model, answer_dir)
    except Exception as e:
        print(f"Failed to update stores with data from {target} : {e}")

    return

"""
async def add_documents(count: int, offset: int):

    global question_store, answer_store

    question_dir = "/" + TMP + QUESTION_STORE_DIR
    answer_dir = "/" + TMP + ANSWER_STORE_DIR

    s3_client = S3Client()

    pq_ids = await get_pq_ids()
    embed_model = OpenAIEmbeddings(
                       model="text-embedding-3-small",
                    )

    # batches of 1 to avoid silly exclusions
    for i in range(offset, count + offset):
        print(f"Retrieving index {i}")
        questions, not_retrieved_ids = get_specific_question_details([pq_ids[i]])
        if questions:
            try:
                df = pd.DataFrame(questions)
                df = populate_embeddable_questions(df)
                df = populate_embeddable_answers(df)
            except Exception as e:
                print(f"Failed to set Dataframe for questions {questions} : {e}")


            try:
                question_documents, answer_documents, success_ids, failed_ids = create_documents(df)
                question_store = update_vector_store(s3_client, question_documents, embed_model, question_dir)
                answer_store = update_vector_store(s3_client, answer_documents, embed_model, answer_dir)
            except Exception as e:
                print(f"Failed to update stores with {questions} : {e}")
"""


async def get_pq_ids():
    global pq_ids

    # hack to overcome ruff insistence on avoiding /tmp
    store_dir = "/" + TMP + QUESTION_STORE_DIR
    pq_ids_file = store_dir + IDS_FILE

    s3_client = S3Client()

    exists = s3_client.check_object_existence(pq_ids_file)

    if not exists:
        print("Retrieving ids")
        try:
            pq_ids = get_all_question_ids(answering_body_id=13, house="Commons")
            print(f"Retrieved {len(pq_ids)} PQs from parliament api")
        except Exception as e:
            print(f"Error retrieving ids {e}")

        create_directory_if_necessary(store_dir)

        try:
            print(f"Writing csv file of ids {pq_ids_file}")
            with open(pq_ids_file, "w") as csvfile:
                writer = csv.writer(csvfile)
                for pid in pq_ids:
                    writer.writerow([pid])
        except Exception as e:
            print(f"Error storing ids in file {pq_ids_file}: {e}")
        try:
            print("Storing id file in S3")

            s3_client.upload_file(pq_ids_file)
        except Exception as e:
            print(f"Error storing  {pq_ids_file} in S3: {e}")

    else:
        print(f"Checking for ids directory {store_dir} before download of ids file")
        create_directory_if_necessary(store_dir)

        try:
            pq_ids = []
            print(f"Downloading Ids file {pq_ids_file}")
            s3_client.download_file(pq_ids_file, pq_ids_file)

            with open(pq_ids_file) as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    pq_ids.append(int(row[0]))
            print(f"Read {len(pq_ids)} PQ ids from file.")
        except Exception as e:
            print(f"Error downloading/reading {pq_ids_file} from S3: {e}")

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
       print("Vector stores not located")
    else:
        print("Retrieving stores")

        question_store = load_store(s3_client, question_dir, embed_model)
        answer_store = load_store(s3_client, answer_dir, embed_model)

    return question_store, answer_store


async def store_documents(s3_client, embed_model, question_dir, answer_dir):

    global question_store, answer_store

    print("Retrieving documents for storage")

    pq_ids = await get_pq_ids()
    if len(pq_ids) > 0:
        print(f"First PQ id {pq_ids[0]}")

    questions, not_retrieved_ids = get_specific_question_details(pq_ids)

    # use Pandas for text manipulation
    if questions:
        try:
            df = pd.DataFrame(questions)
            df = populate_embeddable_questions(df)
        except Exception as e:
            print(f"Failed to set Dataframe for questions {questions}:\n{e}")
            return None

    try:
        question_documents, answer_documents, success_ids, failed_ids = create_documents(df)
        if failed_ids:
            print(f"The following policy ids were not stored:\n{failed_ids}")

        question_store = update_vector_store(s3_client, question_documents, embed_model, question_dir)
        answer_store = update_vector_store(s3_client, answer_documents, embed_model, answer_dir)
    except Exception as e:
        print(f"Failed to update stores {e}")

    return question_store, answer_store


def get_question_match(question, limit):
    print(question_store)
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

def store_output(filename, json_content):
    s3_client = S3Client()
    # hack to overcome ruff insistence on avoiding /tmp

    store_dir = "/" + TMP + QUESTION_STORE_DIR
    create_directory_if_necessary(store_dir)

    target =  store_dir + filename

    with open(target, "w") as f:
        json.dump(json_content, f)

    s3_client.upload_file(target)


def read_output(filename):
    s3_client = S3Client()
    # hack to overcome ruff insistence on avoiding /tmp
    store_dir = "/" + TMP + QUESTION_STORE_DIR

    target =  store_dir + filename

    exists = s3_client.check_object_existence(target)

    if not exists:
       return {"message":"Semantic Chat output not yet available, please try again soon."}

    create_directory_if_necessary(store_dir)

    s3_client.download_file(target, target)

    with open(target) as file:
        result = json.load(file)

    print(f"Successfully read {result}")
    return result

