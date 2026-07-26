from typing import Protocol, Callable
import json

from openai import OpenAI
from minsearch import Index


class BaseRetriever(Protocol):
    def search(self, query: str) -> list[dict]:
        ...


class BaseLLM(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, str]] = []
    ) -> str:
        ...


class MinsearchRetriever:
    def __init__(
        self,
        index: Index,
        boost_dict: dict[str, float] = None,
        filter_dict: dict[str, str] = None
    ):
        self.index = index
        self.boost_dict = boost_dict
        self.filter_dict = filter_dict

    def search(self, query: str, num_results: int = 5) -> list[dict[str, str]]:
        return self.index.search(
            query=query,
            num_results=num_results,
            boost_dict=self.boost_dict,
            filter_dict=self.filter_dict
        )
    

class MessageHistory:
    def __init__(self, instructions: str | None):
        self.messages: list[dict[str, str]] = []
        if instructions:
            self.messages.append({"role": "developer", "content": instructions})
    
    def add_prompt(self, role: str, prompt: str) -> None:
        self.messages.append({"role": role, "content": prompt})
    
    def append(self, messages: dict[str, str]) -> None: 
        self.messages.append(messages)

    def extend(self, messages: list[dict[str, str]]) -> None:
        self.messages.extend(messages)
    

class OpenAILLM:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def complete(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, str]] = []
    ) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=messages,
            tools=tools
        )
        return response
    
    
def format_prompt(query: str, search_results: list[dict[str, str]]):
    lines = []
    for doc in search_results:
        keys = doc.keys()
        for key in keys:
            lines.append(f"{key}: {doc[key]}")
        lines.append("")
    context = "\n".join(lines).strip()

    prompt_template = """
QUESTION: {query}

CONTEXT:
{context}
""".strip()
    return prompt_template.format(query=query, context=context)


def rag_pipeline(
    query: str,
    retriever: BaseRetriever,
    message_history: MessageHistory,
    llm: BaseLLM
):
    search_results = retriever.search(query=query)
    message = format_prompt(query=query, search_results=search_results)
    message_history.add_prompt(role="user", prompt=message)
    response = llm.complete(message_history.messages)
    return response


def tool_calls(call, tool_registry: dict[str, Callable]):
    args = json.loads(call.arguments)
    func = tool_registry[call.name]
    result = func(**args)
    result_json = json.dumps(result)
    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": result_json,
    }   


def agentic_loop(
    query: str,
    tools: list,
    tool_registry: dict[str, Callable],
    message_history: MessageHistory,
    llm: BaseLLM
):
    it = 0
    message_history.add_prompt(role="user", prompt=query)
    while True:
        print(f"Iteration #{it}...")
        has_function_calls = False

        response = llm.complete(messages=message_history.messages, tools=tools)
        message_history.extend(response.output)

        for output in response.output:
            if output.type == "function_call":
                print(f"function_call: {output.name}, {output.arguments}")
                call_output = tool_calls(call=output, tool_registry=tool_registry)
                message_history.append(call_output)
                has_function_calls = True
        
        it += 1
        if has_function_calls == False:
            break
        
    return response