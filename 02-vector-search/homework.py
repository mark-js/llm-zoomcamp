import numpy as np

from embedder import Embedder
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index, VectorSearch


def ingest() -> list[dict[str, str]]:
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    return [file.parse() for file in reader.read()]


def fit_vector_search(X, documents):
    index = VectorSearch(
        keyword_fields=["filename"]
    )
    index.fit(X, documents)
    return index


def fit_index(documents):
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"]
    )
    index.fit(documents)
    return index


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


def main():
    embedder = Embedder()
    v = embedder.encode("How does approximate nearest neighbor search work?")
    print(v[0])

    documents = ingest()
    filename = "02-vector-search/lessons/07-sqlitesearch-vector.md"
    contents = next(
        (page["content"] for page in documents if page["filename"] == filename), None
    )
    contents_v = embedder.encode(contents)
    cosine_similarity = contents_v.dot(v)
    print(cosine_similarity)

    chunks = chunk_documents(documents=documents, size=2000, step=1000)
    chunks_content = [chunk['content'] for chunk in chunks]
    X = embedder.encode_batch(chunks_content)
    scores = X.dot(v)
    idx = np.argmax(scores)
    print(chunks[idx]["filename"])

    query_v = embedder.encode("What metric do we use to evaluate a search engine?")
    vindex = fit_vector_search(X, chunks)
    result = vindex.search(query_v)
    print(result[0]["filename"])

    query = "How do I store vectors in PostgreSQL?"
    query_v = embedder.encode(query)
    index = fit_index(chunks)
    result_keyword = index.search(query=query, num_results=5)
    result_vector = vindex.search(query_v, num_results=5)
    print([i["filename"] for i in result_keyword])
    print([i["filename"] for i in result_vector])

    query = "How do I give the model access to tools?"
    query_v = embedder.encode(query)
    index = fit_index(chunks)
    result_keyword = index.search(query=query, num_results=5)
    result_vector = vindex.search(query_v, num_results=5)
    print([i["filename"] for i in result_keyword])
    print([i["filename"] for i in result_vector])
    results = rrf([result_keyword, result_vector])
    results


    












if __name__ == "__main__":
    main()
