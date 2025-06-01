import os
from logging import getLogger

import pandas as pd
from fastapi import APIRouter
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from app.config import config as settings

from .policy_retrieval import get_all_question_details

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
    print(settings.OPENAI_API_KEY)
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    question_documents = []
    answer_documents = []

    for _index, question in df.iterrows():
        print(f"Creating embedding for {question}")
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
    questions = get_all_question_details(answering_body_id)
    # use Pandas for text manipulation
    df = pd.DataFrame(questions)
    df = populate_embeddable_questions(df)
    df = populate_embeddable_answers(df)

    question_documents, answer_documents = create_documents(df)
    question_store = create_vector_store(question_documents)
    answer_store = create_vector_store(answer_documents)
    return question_store, answer_store
