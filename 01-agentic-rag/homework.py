from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index
from openai import OpenAI
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag import MinsearchRetriever, MessageHistory, OpenAILLM, rag_pipeline, agentic_loop

INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

INSTRUCTIONS_AGENT = """
You're a course teaching assistant.
Answer the student's question using the search tool.
Make multiple searches with different keywords before answering.
"""


search_tool = {
    "type": "function",
    "name": "search",
    "description": "Search the course documents for entries matching the given query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the course documents."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}


class LLMOpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str
    port: int
    model: str | None = None
    api_key: SecretStr | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def ingest() -> list[dict[str, str]]:
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    files = reader.read()

    documents = []
    for file in files:
        doc = file.parse()
        documents.append(doc)
    return documents


def fit_index(documents):
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"]
    )
    index.fit(documents)
    return index


def main():
    settings = LLMOpenAISettings()

    documents = ingest()
    chunks = chunk_documents(documents=documents, size=2000, step=1000)
    index = fit_index(documents=chunks)

    retriever = MinsearchRetriever(index=index)

    message_history = MessageHistory(instructions=INSTRUCTIONS_AGENT)

    client = OpenAI(base_url=settings.base_url, api_key=settings.api_key.get_secret_value())
    llm = OpenAILLM(client, model=settings.model)

    # query = "How does the agentic loop keep calling the model until it stops?"
    # response = rag_pipeline(query=query, retriever=retriever, message_history=message_history, llm=llm)
    # response

    tool_registry = {
        "search": retriever.search
    }

    query = "How does the agentic loop work, and how is it different from plain RAG?"
    response = agentic_loop(
        query=query,
        tools=[search_tool],
        tool_registry=tool_registry,
        message_history=message_history,
        llm=llm
    )
    response


if __name__ == "__main__":
    main()