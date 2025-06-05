import os
from logging import getLogger
from pathlib import Path

import pandas as pd
from fastapi import APIRouter
from langchain.vectorstores import FAISS
from langchain.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.config import config as settings

from .policy_retrieval import get_all_question_details

QUESTION_STORE_FILE="question_store"
ANSWER_STORE_FILE="answer_store"

question_store = None
answer_store = None

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


def create_vector_store(documents, embed_model, store_path):
    print(f"Would store here {store_path}")

    vector_store = FAISS.from_documents(
                                      documents,
                                      embedding=embed_model,
                                      distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
                                      )
    # Save vector store
    vector_store.save_local(store_path)
    for file in store_path.iterdir():
        print(file)

    return vector_store


def load_store(store_path, embed_model):
    # Load FAISS index back
    print("LOADING")
    for file in store_path.iterdir():
        print(file)

    return FAISS.load_local(store_path, embed_model,allow_dangerous_deserialization=True)


async def check_storage():

    global question_store, answer_store

    parts = ["tmp","questions","answers"]
    question_path = Path("/" + parts[0] + "/" + parts[1] + "/")
    answer_path = Path("/" + parts[0] + "/" + parts[2] + "/")
    embed_model = OpenAIEmbeddings(
                       model="text-embedding-3-small",
                 )
    if (not question_path.exists() or
        not answer_path.exists()):
       print("STORING")
       await store_documents(embed_model, question_path, answer_path)
    else:
        print("Retrieving stores")

    question_store = load_store(question_path, embed_model)
    answer_store = load_store(answer_path, embed_model)

    return question_store, answer_store


async def store_documents(embed_model, question_path, answer_path,answering_body_id=13):
    print("Retrieving documents for storage")
    questions = get_all_question_details(answering_body_id)
    # use Pandas for text manipulation
    df = pd.DataFrame(questions)
    df = populate_embeddable_questions(df)
    df = populate_embeddable_answers(df)

    # temp storage for checkpoint
    pq_path = Path(question_path , "pq.csv")
    df.to_csv(pq_path)

    question_documents, answer_documents = create_documents(df)
    question_store = create_vector_store(question_documents, embed_model, question_path)
    answer_store = create_vector_store(answer_documents, embed_model, answer_path)

    return question_store, answer_store


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
