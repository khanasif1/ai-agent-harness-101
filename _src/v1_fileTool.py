import sys

sys.stdout.reconfigure(encoding="utf-8")  # so emoji don't crash the Windows console

import json
from pathlib import Path

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI
from openai.types.responses import ResponseInputParam, ToolParam

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)
client = OpenAI(
    base_url="https://use-ai-foundry-demo.services.ai.azure.com/openai/v1/",
    api_key=token_provider,
)
MODEL = "gpt-6-astra"

WORKSPACE = Path(__file__).parent / "workspace"


# --- the actual tools: plain Python functions -------------------------------

def list_files() -> str:
    """List the files in the assistant's workspace folder."""
    return "\n".join(p.name for p in WORKSPACE.iterdir()) or "(empty)"


def read_file(filename: str) -> str:
    """Read a file from the workspace folder."""
    path = WORKSPACE / filename
    if not path.is_file():
        return f"error: no file named {filename}"
    return path.read_text(encoding="utf-8")


TOOLS = {"list_files": list_files, "read_file": read_file}

# --- what the model sees: a JSON description of each tool -------------------

TOOL_SCHEMAS: list[ToolParam] = [
    {
        "type": "function",
        "name": "list_files",
        "description": "List the files in the user's workspace folder.",
        "parameters": {"type": "object", "properties": {}},
        "strict": False,
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "Read one file from the user's workspace folder.",
        "parameters": {
            "type": "object",
            "properties": {"filename": {"type": "string"}},
            "required": ["filename"],
        },
        "strict": False,
    },
]


def chat(user_message: str) -> str:
    messages: ResponseInputParam = [
        {"role": "system", "content": "You are a helpful personal assistant."},
        {"role": "user", "content": user_message},
    ]

    response = client.responses.create(
        model=MODEL, input=messages, tools=TOOL_SCHEMAS
    )
    tool_calls = [item for item in response.output if item.type == "function_call"]

    # The model didn't want a tool - just answer.
    if not tool_calls:
        return response.output_text

    # Run the tool it asked for and give it the result... ONCE.
    # Replay all output items, including reasoning, before the tool results.
    messages.extend(response.model_dump(exclude_unset=True)["output"])
    for call in tool_calls:
        args = json.loads(call.arguments or "{}")
        print(f"  [tool] {call.name}({args})")
        result = TOOLS[call.name](**args)
        messages.append(
            {"type": "function_call_output", "call_id": call.call_id, "output": result}
        )

    final = client.responses.create(model=MODEL, input=messages)
    return final.output_text


if __name__ == "__main__":
    print(f"v1 assistant ({MODEL}) - ctrl+c to quit")
    try:
        while True:
            question = input("\nyou: ")
            print("\nassistant:", chat(question))
    except (EOFError, KeyboardInterrupt):
        print("\nbye!")
