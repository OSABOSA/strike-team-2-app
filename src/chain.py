from typing import Callable
from openai import OpenAI
import json

from src.data.callback_handler import CallbackType


class LlmModule:

    def __init__(self, progress_callback: Callable[[CallbackType, str], None], db_query_callback: Callable[[str, int], list], model_name="gpt-4o-mini"):
        self.progress_callback = progress_callback
        self.db_query_callback = db_query_callback
        self.client = OpenAI()
        self.model = model_name
        self.tools = [{
            "type": "function",
            "name": "search_knowledge_base",
            "description": "Query a knowledge base to retrieve relevant info on a topic. Knowledge base contains many reviews of various car models.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query based the user question. This needs to be kept neutral for best results."
                    },
                    "num_results": {
                        "type": "number",
                        "description": "Number of top results to return. Default go-to number is 5. For a more broad search pass higher numbers."
                    }
                },
                "required": [
                    "query",
                    "num_results"
                ],
                "additionalProperties": False
            }
        }]
        self.messages = []  # [{"role": "user", "content": ""}]

    def process_tool_calls(self, response):
        used_tools = False
        for tool_call in response:
            if tool_call.type != "function_call":
                continue
            used_tools = True

            self.messages.append(tool_call)  # broken
            args = json.loads(tool_call.arguments)
            self.progress_callback(CallbackType.STATUS, args["query"])
            result: list = self.db_query_callback(args["query"], args["num_results"])  # database interface

            if len(result) is not 0:
                self.messages.append({
                    "status": "success",
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": json.dumps(result)
                })
            else:
                self.messages.append({
                    "status": "fail",
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": ""
                })
        return used_tools

    def reset_messages(self):
        self.messages = []

    # broken
    def get_response(self):
        stream = self.client.responses.create(
            model=self.model,
            input=self.messages,
            tools=self.tools,
            stream=True
        )
        items = []
        for event in stream:
            if event.type == "response.output_text.delta":
                self.progress_callback(CallbackType.DELTA, event.delta)
            if event.type == "response.output_text.done":
                self.progress_callback(CallbackType.RESPONSE, event.text)
            if event.type == "response.output_item.done":
                items.append(event.item)
        return items

    '''def get_response(self):
        return self.client.responses.create(
            model=self.model,
            input=self.messages,
            tools=self.tools
        )'''


    def chat(self, query):
        self.messages.append({
            "role": "user",
            "content": query
        })
        response = self.get_response()
        if self.process_tool_calls(response):
            response = self.get_response()
            self.messages.append({
                "role": "assistant",
                "content": response[0].content
            })
        else:
            self.messages.append({
                "role": "assistant",
                "content": response[0].content
            })
        self.progress_callback(CallbackType.RESPONSE, response[0].content)
