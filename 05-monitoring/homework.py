from openai import OpenAI
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
import pandas as pd
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import sqlite3

from gitsource import GithubRepositoryDataReader
from minsearch import Index
from rag import MinsearchRetriever, OpenAILLM, MessageHistory, format_prompt
from span_exporter import SQLiteSpanExporter

provider = TracerProvider()
provider.add_span_processor(
    SimpleSpanProcessor(SQLiteSpanExporter("data/traces.db"))
)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-zoomcamp")


INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don"t know."
"""


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


def rag_pipeline_traced(
    query: str,
    llm: BaseLLM,
    retriever: BaseRetriever,
    message_history: MessageHistory
):
    with tracer.start_as_current_span("retriever.search") as span:
        search_results = retriever.search(query=query)

    message = format_prompt(query=query, search_results=search_results)
    message_history.add_prompt(role="user", prompt=message)

    with tracer.start_as_current_span("llm.create") as span:
        response = llm.create(message_history.messages)

    return response
   

def main():

    settings = LLMOpenAISettings()
    client = OpenAI(base_url=settings.base_url, api_key=settings.api_key.get_secret_value())
    llm = OpenAILLM(client=client, model=settings.model)

    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    documents = [file.parse() for file in reader.read()]
    index = Index(text_fields=["content"], keyword_fields=["filename"])
    index.fit(documents)

    search = MinsearchRetriever(index=index)

    message_history = MessageHistory(instructions=INSTRUCTIONS) 

    with tracer.start_as_current_span("rag_pipeline_traced") as span:
        response = rag_pipeline_traced(
            query="How does the agentic loop keep calling the model until it stops?",
            llm=llm,
            retriever=search,
            message_history=message_history
        )
        span.set_attribute("input_tokens", response.usage.prompt_tokens)
        span.set_attribute("output_tokens", response.usage.completion_tokens)

    # Question 5
    query = """SELECT
    name,
    (end_time - start_time) / 1e9 AS duration_s
FROM
    spans
WHERE
    name != 'llm.create'"""

    con = sqlite3.connect("data/traces.db")
    cur = con.cursor()
    df = pd.read_sql(query, con)
    print(df)

    # Question 6
    query = """SELECT
    name,
    (end_time - start_time) / 1e9 AS duration_s
FROM
    spans
WHERE
    name = 'llm.create'"""

    df = pd.read_sql(query, con)
    print(df)

    con.close()


if __name__ == "__main__":
    main()