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
MEMORY_FILE = Path(__file__).parent / "memory.md"


# --- memory: a markdown file, that's it -------------------------------------

def load_memory() -> str:
    if MEMORY_FILE.is_file():
        return MEMORY_FILE.read_text(encoding="utf-8")
    return "(nothing saved yet)"


def save_memory(fact: str) -> str:
    """Append one fact about the user to long-term memory."""
    with MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- {fact}\n")
    return f"saved: {fact}"


# --- tools ------------------------------------------------------------------

def list_files() -> str:
    return "\n".join(p.name for p in WORKSPACE.iterdir()) or "(empty)"


def read_file(filename: str) -> str:
    path = WORKSPACE / filename
    return path.read_text(encoding="utf-8") if path.is_file() else f"error: no {filename}"


TOOLS = {"list_files": list_files, "read_file": read_file, "save_memory": save_memory}

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
    {
        "type": "function",
        "name": "save_memory",
        "description": (
            "Save one short fact about the user to long-term memory. "
            "Use whenever you learn something worth remembering: their "
            "name, preferences, projects, recurring tasks."
        ),
        "parameters": {
            "type": "object",
            "properties": {"fact": {"type": "string"}},
            "required": ["fact"],
        },
        "strict": False,
    },
]


SYSTEM_PROMPT = """You are a helpful personal assistant.

Here is what you remember about the user from previous sessions:
{memory}

When you learn a new lasting fact about the user, save it with save_memory."""


def run_agent(messages: ResponseInputParam) -> str:
    while True:
        response = client.responses.create(
            model=MODEL, input=messages, tools=TOOL_SCHEMAS
        )
        # Keep final answers, reasoning, and function calls for subsequent turns.
        messages.extend(response.model_dump(exclude_unset=True)["output"])
        tool_calls = [item for item in response.output if item.type == "function_call"]
        if not tool_calls:
            return response.output_text

        for call in tool_calls:
            args = json.loads(call.arguments or "{}")
            print(f"  [tool] {call.name}({args})")
            result = TOOLS[call.name](**args)
            messages.append(
                {"type": "function_call_output", "call_id": call.call_id, "output": result}
            )


if __name__ == "__main__":
    # Conversation history persists across turns now, too.
    messages: ResponseInputParam = [
        {"role": "system", "content": SYSTEM_PROMPT.format(memory=load_memory())}
    ]
    print(f"v4 assistant ({MODEL}) - memory loaded - ctrl+c to quit")
    try:
        while True:
            messages.append({"role": "user", "content": input("\nyou: ")})
            print("\nassistant:", run_agent(messages))
    except (EOFError, KeyboardInterrupt):
        print("\nbye!")
