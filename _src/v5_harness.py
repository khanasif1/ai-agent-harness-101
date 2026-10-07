import sys

sys.stdout.reconfigure(encoding="utf-8")  # so emoji don't crash the Windows console

import asyncio
import json
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Literal, cast

from anthropic import AsyncAnthropicFoundry
from anthropic.types import ContentBlock, ContentBlockParam, MessageParam, ToolParam
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.responses import (
    ResponseInputItemParam,
    ResponseInputParam,
    ResponseOutputItem,
    ToolParam as ResponseToolParam,
)
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
WORKSPACE = ROOT / "workspace"
MEMORY_FILE = ROOT / "memory.md"
MCP_CONFIG = ROOT / "mcp_servers.json"


# ---------------------------------------------------------------------------
# Pillar 3: Foundry deployments, authenticated with Microsoft Entra ID.
# ---------------------------------------------------------------------------

PROJECT_ENDPOINT = (
    "https://use-ai-foundry-demo.services.ai.azure.com/api/projects/proj-default"
)
RESOURCE_ENDPOINT = PROJECT_ENDPOINT.split("/api/projects/", 1)[0]


class ModelConfig(BaseModel):
    base_url: str
    model: str
    api: Literal["responses", "chat_completions", "anthropic"] = "responses"


MODELS: dict[str, ModelConfig] = {
    "gpt": ModelConfig(
        base_url=f"{PROJECT_ENDPOINT}/openai/v1/",
        model="gpt-6-astra",
    ),
    "claude": ModelConfig(
        base_url=f"{RESOURCE_ENDPOINT}/anthropic/",
        model="claude-sonnet-5",
        api="anthropic",
    ),
    "kimi": ModelConfig(
        base_url=f"{PROJECT_ENDPOINT}/openai/v1/",
        model="Kimi-K2.6",
        api="chat_completions",
    ),
}

model_name = "gpt"  # switch at runtime with /model

credential = DefaultAzureCredential()
sync_token_provider = get_bearer_token_provider(
    credential, "https://ai.azure.com/.default"
)


async def token_provider() -> str:
    return await asyncio.to_thread(sync_token_provider)


clients: dict[str, AsyncOpenAI | AsyncAnthropicFoundry] = {
    name: (
        AsyncAnthropicFoundry(
            base_url=cfg.base_url, azure_ad_token_provider=token_provider
        )
        if cfg.api == "anthropic"
        else AsyncOpenAI(base_url=cfg.base_url, api_key=token_provider)
    )
    for name, cfg in MODELS.items()
}


# ---------------------------------------------------------------------------
# Pillar 2: memory. A markdown file in the prompt + a tool to append to it.
# ---------------------------------------------------------------------------

def load_memory() -> str:
    if MEMORY_FILE.is_file():
        return MEMORY_FILE.read_text(encoding="utf-8")
    return "(nothing saved yet)"


def save_memory(fact: str) -> str:
    with MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- {fact}\n")
    return f"saved: {fact}"


# ---------------------------------------------------------------------------
# Local tools (the ones we wrote by hand back in v1)
# ---------------------------------------------------------------------------

def list_files() -> str:
    return "\n".join(p.name for p in WORKSPACE.iterdir()) or "(empty)"


def read_file(filename: str) -> str:
    path = WORKSPACE / filename
    return path.read_text(encoding="utf-8") if path.is_file() else f"error: no {filename}"


def write_file(filename: str, content: str) -> str:
    (WORKSPACE / filename).write_text(content, encoding="utf-8")
    return f"wrote {filename}"


LOCAL_TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "save_memory": save_memory,
}

TOOL_SCHEMAS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files in the user's workspace folder.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one file from the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file in the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": (
                "Save one short fact about the user to long-term memory. "
                "Use whenever you learn something lasting: their name, "
                "preferences, projects, recurring tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"],
            },
        },
    },
]


SYSTEM_PROMPT = """You are a helpful personal assistant running inside a \
custom harness. Be concise.

The user's workspace folder holds their personal files: notes, todo lists, \
ideas. You have tools to list, read, and write those files. Never claim you \
lack access to the user's files or tasks - use your tools to look.

Here is what you remember about the user from previous sessions:
{memory}

When you learn a new lasting fact about the user, save it with save_memory."""

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str = ""
    reasoning_content: str | None = None
    claude_content: list[ContentBlock] | None = None
    gpt_output: list[ResponseOutputItem] | None = None


# Keep provider-specific reasoning for replay, but never send it to another provider.
class ReasoningMessageParam(ChatCompletionAssistantMessageParam, total=False):
    reasoning_content: str


messages = [Message(role="system", content=SYSTEM_PROMPT.format(memory=load_memory()))]


# ---------------------------------------------------------------------------
# Pillar 1: MCP (same component we built in v3)
# ---------------------------------------------------------------------------

mcp_sessions: dict[str, ClientSession] = {}  # tool name -> its server session


async def connect_mcp(stack: AsyncExitStack):
    """Launch every server in mcp_servers.json and merge in its tools."""
    config = json.loads(MCP_CONFIG.read_text(encoding="utf-8"))
    index = os.getenv("UV_DEFAULT_INDEX")
    for name, spec in config["mcpServers"].items():
        params = StdioServerParameters(
            command=spec["command"],
            args=spec["args"],
            env={"UV_DEFAULT_INDEX": index} if index else None,
        )
        try:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except Exception as e:
            print(f"  [mcp] '{name}' failed to start: {e}")
            continue

        tools = (await session.list_tools()).tools
        for tool in tools:
            mcp_sessions[tool.name] = session
            TOOL_SCHEMAS.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                }
            )
        print(f"  [mcp] connected '{name}': {[t.name for t in tools]}")


async def call_tool(name: str, args: dict) -> str:
    if name in LOCAL_TOOLS:
        return LOCAL_TOOLS[name](**args)
    if name in mcp_sessions:
        result = await mcp_sessions[name].call_tool(name, args)
        return "\n".join(c.text for c in result.content if hasattr(c, "text"))
    return f"error: unknown tool {name}"


# ---------------------------------------------------------------------------
# Provider adapters and the shared agent loop
# ---------------------------------------------------------------------------

def tool_arguments(call: ToolCall) -> dict[str, object]:
    args = json.loads(call.arguments)
    if not isinstance(args, dict):
        raise ValueError(f"Tool '{call.name}' arguments must be a JSON object.")
    return args


def openai_messages(include_reasoning: bool) -> list[ChatCompletionMessageParam]:
    history: list[ChatCompletionMessageParam] = []
    for message in messages:
        if message.role == "assistant":
            reply: ReasoningMessageParam = {
                "role": "assistant",
                "content": message.content,
            }
            if message.tool_calls:
                reply["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in message.tool_calls
                ]
            if include_reasoning and message.reasoning_content is not None:
                reply["reasoning_content"] = message.reasoning_content
            history.append(reply)
        elif message.role == "tool":
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.content,
                }
            )
        elif message.role == "system":
            history.append({"role": "system", "content": message.content})
        else:
            history.append({"role": "user", "content": message.content})
    return history


def responses_input() -> ResponseInputParam:
    history: ResponseInputParam = []
    for message in messages:
        if message.gpt_output is not None:
            # The Responses API accepts its output items as subsequent input.
            history.extend(
                cast(ResponseInputItemParam, item.model_dump(exclude_unset=True))
                for item in message.gpt_output
            )
        elif message.role == "assistant":
            if message.content:
                history.append(
                    {"type": "message", "role": "assistant", "content": message.content}
                )
            history.extend(
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                for call in message.tool_calls
            )
        elif message.role == "tool":
            history.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
            )
        else:
            history.append(
                {"type": "message", "role": message.role, "content": message.content}
            )
    return history


def responses_tools() -> list[ResponseToolParam]:
    return [
        {
            "type": "function",
            "name": schema["function"]["name"],
            "description": schema["function"].get("description") or "",
            "parameters": schema["function"].get("parameters", {}),
            "strict": False,
        }
        for schema in TOOL_SCHEMAS
    ]


def claude_messages() -> list[MessageParam]:
    history: list[MessageParam] = []
    results: list[ContentBlockParam] = []
    tool_ids: dict[str, str] = {}
    for index, message in enumerate(messages[1:], start=1):
        if message.role == "tool":
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_ids[message.tool_call_id],
                    "content": message.content,
                }
            )
            continue
        if results:
            history.append({"role": "user", "content": results})
            results = []
        if message.role == "assistant":
            blocks: list[ContentBlockParam | ContentBlock] = []
            if message.claude_content is not None:
                blocks.extend(message.claude_content)
                tool_ids = {call.id: call.id for call in message.tool_calls}
            else:
                # Kimi IDs can contain "." and ":", which Claude rejects.
                tool_ids = {
                    call.id: f"toolu_{index}_{number}"
                    for number, call in enumerate(message.tool_calls)
                }
                if message.content:
                    blocks.append({"type": "text", "text": message.content})
                blocks.extend(
                    {
                        "type": "tool_use",
                        "id": tool_ids[call.id],
                        "name": call.name,
                        "input": tool_arguments(call),
                    }
                    for call in message.tool_calls
                )
            history.append({"role": "assistant", "content": blocks})
        elif message.role == "user":
            history.append({"role": "user", "content": message.content})
        else:
            raise ValueError("Claude history must have only one initial system message.")
    if results:
        history.append({"role": "user", "content": results})
    return history


def claude_tools() -> list[ToolParam]:
    tools: list[ToolParam] = []
    for schema in TOOL_SCHEMAS:
        function = schema["function"]
        tools.append(
            {
                "name": function["name"],
                "description": function.get("description") or "",
                "input_schema": {"type": "object", **function.get("parameters", {})},
            }
        )
    return tools


async def next_message(
    cfg: ModelConfig, client: AsyncOpenAI | AsyncAnthropicFoundry
) -> Message:
    if isinstance(client, AsyncAnthropicFoundry):
        response = await client.messages.create(
            model=cfg.model,
            system=messages[0].content,
            messages=claude_messages(),
            tools=claude_tools(),
            max_tokens=16384,
        )
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"{cfg.model} reached its output token limit.")
        return Message(
            role="assistant",
            content="\n".join(
                block.text for block in response.content if block.type == "text"
            ),
            tool_calls=[
                ToolCall(id=block.id, name=block.name, arguments=json.dumps(block.input))
                for block in response.content
                if block.type == "tool_use"
            ],
            claude_content=response.content,
        )
    elif cfg.api == "responses":
        response = await client.responses.create(
            model=cfg.model, input=responses_input(), tools=responses_tools()
        )
        if response.status != "completed":
            raise RuntimeError(
                f"{cfg.model} response did not complete: "
                f"{response.status}; {response.incomplete_details or response.error}"
            )
        return Message(
            role="assistant",
            content=response.output_text,
            tool_calls=[
                ToolCall(id=item.call_id, name=item.name, arguments=item.arguments)
                for item in response.output
                if item.type == "function_call"
            ],
            gpt_output=response.output,
        )
    else:
        response = await client.chat.completions.create(
            model=cfg.model,
            messages=openai_messages(include_reasoning=cfg.model == MODELS["kimi"].model),
            tools=TOOL_SCHEMAS,
        )
        if not response.choices:
            raise RuntimeError(f"{cfg.model} returned no completion choices.")
        choice = response.choices[0]
        if choice.finish_reason in ("length", "content_filter"):
            raise RuntimeError(f"{cfg.model} stopped before completion: {choice.finish_reason}")
        message = choice.message
        calls = []
        for call in message.tool_calls or []:
            if call.type != "function":
                raise ValueError(f"Unsupported tool call type: {call.type}")
            calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
            )
        reasoning = (message.model_extra or {}).get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError(f"{cfg.model} returned non-text reasoning_content.")
        return Message(
            role="assistant",
            content=message.content or message.refusal or "",
            tool_calls=calls,
            reasoning_content=reasoning,
        )


async def run_agent(user_message: str) -> str:
    cfg = MODELS[model_name]
    client = clients[model_name]
    messages.append(Message(role="user", content=user_message))
    while True:
        message = await next_message(cfg, client)
        if not message.content and not message.tool_calls:
            raise RuntimeError(f"{cfg.model} returned no text or tool calls.")
        messages.append(message)
        if not message.tool_calls:
            return message.content
        for call in message.tool_calls:
            args = tool_arguments(call)
            print(f"  [tool] {call.name}({args})")
            result = await call_tool(call.name, args)
            messages.append(
                Message(role="tool", tool_call_id=call.id, content=result)
            )


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

def handle_command(line: str) -> bool:
    """Returns True if the line was a command."""
    global model_name
    if not line.startswith("/"):
        return False
    cmd, _, arg = line.partition(" ")
    if cmd == "/models":
        for name, cfg in MODELS.items():
            marker = "*" if name == model_name else " "
            print(f" {marker} {name:<12} {cfg.model}  ({cfg.base_url})")
    elif cmd == "/model":
        if arg in MODELS:
            model_name = arg
            print(f"  switched to {arg} ({MODELS[arg].model})")
        else:
            print(f"  unknown model '{arg}' - try /models")
    elif cmd == "/tools":
        for schema in TOOL_SCHEMAS:
            fn = schema["function"]
            origin = "local" if fn["name"] in LOCAL_TOOLS else "mcp"
            print(f"  [{origin}] {fn['name']}: {(fn.get('description') or '')[:60]}")
    elif cmd == "/memory":
        print(load_memory())
    elif cmd == "/quit":
        raise SystemExit
    else:
        print("  commands: /models /model <name> /tools /memory /quit")
    return True


async def main():
    async with AsyncExitStack() as stack:
        stack.enter_context(credential)
        for client in clients.values():
            await stack.enter_async_context(client)
        print("starting harness...")
        await connect_mcp(stack)
        print(
            f"\nharness ready - model: {model_name} "
            f"({MODELS[model_name].model}), "
            f"{len(TOOL_SCHEMAS)} tools, memory loaded"
        )
        print("type /models, /model <name>, /tools, /memory, or just talk\n")
        try:
            while True:
                line = (await asyncio.to_thread(input, "you: ")).strip()
                if not line or handle_command(line):
                    continue
                print("\nassistant:", await run_agent(line), "\n")
        except (EOFError, KeyboardInterrupt, SystemExit):
            print("\nbye!")


if __name__ == "__main__":
    asyncio.run(main())
