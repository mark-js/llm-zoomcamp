import json

from gitsource import GithubRepositoryDataReader, chunk_documents
from openai import OpenAI
from minsearch import Index, VectorSearch
import numpy as np
import pandas as pd
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm.auto import tqdm

from embedder import Embedder


DATA_GEN_INSTRUCTIONS = """
    You emulate a student who is taking our LLM course.
    You are given one lesson page from the course.
    Formulate 5 questions this student might ask that are answered by this page.
    Output the 5 questions in JSON format.

    Rules:
    - The page should contain the answer to each question.
    - Make the questions complete and not too short.
    - Use as few words as possible from the page; don't copy its phrasing.
    - The questions should resemble how people actually ask things online:
    not too formal, not too short, not too long.
    - Ask about the content of the lesson, not about its formatting or filename.

    Example JSON output:
    {
        "questions": [
            <question1>,
            <question2>,
            <question3>,
            <question4>,
            <question5>
        ]
    }
    """.strip()


class LLMOpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: str
    model: str | None = None
    api_key: SecretStr | None = None


class Questions(BaseModel):
    questions: list[str]


def get_documents():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    return [file.parse() for file in reader.read()]
    


def llm_structured(client, instructions, user_prompt, output_type, model="deepseek-v4-flash"):
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"}
    )
    
    output_parsed = Questions.model_validate_json(response.choices[0].message.content)

    return output_parsed, response


def generating_questions():
    settings = LLMOpenAISettings()
    client = OpenAI(base_url=settings.base_url, api_key=settings.api_key.get_secret_value())
    
    documents = get_documents() 

    input_tokens = []
    for i in range(3):
        user_prompt = json.dumps(documents[i])

        output_parsed, response = llm_structured(
            client=client,
            instructions=DATA_GEN_INSTRUCTIONS,
            user_prompt=user_prompt,
            output_type=Questions,
            model=settings.model
        )
        input_tokens.append(response.usage.prompt_tokens)

    average_input_tokens = sum(input_tokens)/len(input_tokens)
    return average_input_tokens


def fit_index(documents):
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"]
    )
    index.fit(documents)
    return index


def text_search(query, index, num_results=5):
    return index.search(query=query, num_results=num_results)


def fit_vector_search(X, documents):
    index = VectorSearch(
        keyword_fields=["filename"]
    )
    index.fit(X, documents)
    return index


def vector_search(query, vindex, embedder, num_results=5):
    query_vector = embedder.encode(query)
    return vindex.search(query_vector=query_vector, num_results=num_results)


def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


def hybrid_search(query, index, vindex, embedder, k=60):
    text_results = text_search(query, index, num_results=10)
    vector_results = vector_search(query, vindex, embedder, num_results=10)
    return rrf([text_results, vector_results], k=k)


def compute_relevance_text(ground_truth, index):
    relevance_total = []
    for _, row in tqdm(ground_truth.iterrows()):
        filename_ground_truth = row["filename"]
        results = text_search(row["question"], index)

        relevance = []
        for result in results:
            relevance.append(int(result["filename"] == filename_ground_truth))

        relevance_total.append(relevance)

    return relevance_total


def compute_relevance_vector(ground_truth, vindex, embedder):
    relevance_total = []
    for _, row in tqdm(ground_truth.iterrows()):
        filename_ground_truth = row["filename"]
        results = vector_search(row["question"], vindex, embedder)

        relevance = []
        for result in results:
            relevance.append(int(result["filename"] == filename_ground_truth))

        relevance_total.append(relevance)

    return relevance_total


def compute_relevance_hybrid(ground_truth, index, vindex, embedder, k=50):
    relevance_total = []
    for _, row in tqdm(ground_truth.iterrows()):
        filename_ground_truth = row["filename"]
        results = hybrid_search(row["question"], index, vindex, embedder, k)

        relevance = []
        for result in results:
            relevance.append(int(result["filename"] == filename_ground_truth))

        relevance_total.append(relevance)

    return relevance_total


def hit_rate(relevance):
    arr = np.asarray(relevance)
    return arr.any(axis=1).mean()


def mrr(relevance):
    arr = np.asarray(relevance)
    arr = arr[arr.any(axis=1)]
    pos = np.argmax(arr, axis=1)
    factor = 1 / (pos + 1)
    return np.sum(factor)/len(relevance)



def main():
    ground_truth = pd.read_csv("data/ground-truth.csv")
    documents = get_documents() 
    chunks = chunk_documents(documents, size=2000, step=1000)

    index = fit_index(chunks)
    
    embedder = Embedder()
    chunks_content = [chunk['content'] for chunk in chunks]
    X = embedder.encode_batch(chunks_content)
    vindex = fit_vector_search(X, chunks)

    query = ground_truth.at[0, "question"]

    text_results = text_search(query, index)
    print(text_results[0]["filename"])

    vector_results = vector_search(query, vindex, embedder)
    print(vector_results[0]["filename"])

    text_relevance = compute_relevance_text(ground_truth, index)
    print(hit_rate(text_relevance))

    vector_relevance = compute_relevance_vector(ground_truth, vindex, embedder)
    print(mrr(vector_relevance))

    ks = [1, 50, 100, 200]

    for k in ks:
        hybrid_relevance = compute_relevance_hybrid(ground_truth, index, vindex, embedder, k)
        print(f"k: {k} - MRR: {mrr(hybrid_relevance)}")

    


    

main()
