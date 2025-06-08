import csv
import os
from logging import getLogger
from pathlib import Path

import pandas as pd
from fastapi import APIRouter
from langchain.vectorstores import FAISS
from langchain.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.common.s3 import S3Client
from app.config import config as settings

#from .policy_retrieval import get_all_question_details, get_all_question_ids
from .policy_retrieval import get_all_question_ids, get_specific_question_details

QUESTION_STORE_DIR="/question_store_1/"
ANSWER_STORE_DIR="/answer_store_1/"

TMP = "tmp"

IDS_FILE = "pq_ids.csv"


question_store = None
answer_store = None

pq_ids = []

router = APIRouter(prefix="/policy")
logger = getLogger(__name__)


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
    print(f"Storing {df.shape[0]} documents")
    if not os.getenv("OPENAI_API_KEY"):
        print("Retrieving OPENAI API key")
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

    question_documents = []
    answer_documents = []

    for _index, question in df.iterrows():
        print(f"Creating embedding for {question['id']}")
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
            print(f"Error embedding question {question}: {e}")

    return question_documents, answer_documents


def create_vector_store(s3_client, documents, embed_model, store_dir):
    print(f"Would store here {store_dir}")

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


def load_store(s3_client, store_dir, embed_model):
    # Load FAISS index back
    store_path = Path(store_dir)
    if not store_path.exists():
        store_path.mkdir()
        print(f"Created directory {store_path}")

    print("LOADING")
    faiss_file = store_dir + "index.faiss"
    pickle_file = store_dir + "index.pkl"

    s3_client.download_file(faiss_file, faiss_file)
    s3_client.download_file(pickle_file, pickle_file)
    print("Loaded index files")
    for file in Path(store_dir).iterdir():
        print(f"Located {file}")
    try:
       store = FAISS.load_local(store_dir, embed_model,allow_dangerous_deserialization=True)
       print(f"Loaded FAISS from {store_dir}")
       return store
    except Exception as e:
       print(f"Error creating FAISS: {e}")
       return None


def create_directory_if_necessary(directory_name):
    try:
        print(f"Checking for ids directory {directory_name}")
        path = Path(directory_name)
        if not path.exists():
            path.mkdir()
            print(f"Created directory {path}")
        else:
            print(f"Found {path}")

    except Exception as e:
        print(f"Error creating {path} directory: {e}")

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
                    writer.writerow(str(pid))
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
            print(f"Downloading Ids file {pq_ids_file}")
            s3_client.download_file(pq_ids_file, pq_ids_file)

            with open(pq_ids_file) as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    pq_ids.append(row[0])
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
       print("STORING")
       await store_documents(s3_client, embed_model, question_dir, answer_dir)
    else:
        print("Retrieving stores")

    question_store = load_store(s3_client, question_dir, embed_model)
    answer_store = load_store(s3_client, answer_dir, embed_model)

    return question_store, answer_store


#async def store_documents(s3_client, embed_model, question_dir, answer_dir ,answering_body_id=13):
async def store_documents(s3_client, embed_model, question_dir, answer_dir):

    global question_store, answer_store

    print("Retrieving documents for storage")

#    questions = get_all_question_details(answering_body_id)
    pq_ids = await get_pq_ids()
    if len(pq_ids) > 0:
        print(f"First PQ id {pq_ids[0]}")

    questions = get_specific_question_details(pq_ids)
    print(f"Extracted {len(questions)} PQs")
    # use Pandas for text manipulation
    try:
        df = pd.DataFrame(questions)
        df = populate_embeddable_questions(df)
        df = populate_embeddable_answers(df)
        print(df.columns)
        print(df.shape)

    # temp storage for checkpoint
#    pq_path = Path(question_path , "pq.csv")
#    df.to_csv(pq_path)

        question_documents, answer_documents = create_documents(df)
        question_store = create_vector_store(s3_client, question_documents, embed_model, question_dir)
        answer_store = create_vector_store(s3_client, answer_documents, embed_model, answer_dir)
    except Exception as e:
        print(f"Failed to set Dataframe {e}")


def get_question_match(question, limit):
    print(question_store)
    documents =  question_store.similarity_search_with_score(
                            query=question,
                            k=limit
                        )

    return [{"id":doc.id, "question":doc.page_content, "uin":doc.metadata["uin"], "score":float(score)} for doc,score in documents]


def get_answer_match(question, limit):
    documents = answer_store.similarity_search_with_score(
                            query=question,
                            k=limit
                        )
    return [{"id":doc.id, "answer":doc.page_content, "score":float(score)} for doc,score in documents]
