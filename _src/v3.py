import sys

sys.stdout.reconfigure(encoding="utf-8")  # so emoji don't crash the Windows console

import asyncio
import json
from contextlib import AsyncExitStack
from pathlib import Path

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from openai.types.responses import ResponseInputParam, ToolParam

credential = DefaultAzureCredential()
sync_token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)


async def token_provider() -> str:
    # Keep synchronous Azure Identity token acquisition off the event loop.
    return await asyncio.to_thread(sync_token_provider)


client = AsyncOpenAI(
    base_url="https://use-ai-foundry-demo.services.ai.azure.com/openai/v1/",
    api_key=token_provider,
)
MODEL = "gpt-6-astra"

WORKSPACE = Path(__file__).parent / "workspace"

# Each server is just a command to launch. Same format Claude Code uses.
MCP_SERVERS = {
    "time": StdioServerParameters(command="uvx", args=["mcp-server-time"]),
    "fetch": StdioServerParameters(command="uvx", args=["mcp-server-fetch"]),
}


# --- our own local tools, unchanged from v2 ---------------------------------

def list_files() -> str:
    return "\n".join(p.name for p in WORKSPACE.iterdir()) or "(empty)"


def read_file(filename: str) -> str:
    path = WORKSPACE / filename
    return path.read_text(encoding="utf-8") if path.is_file() else f"error: no {filename}"


LOCAL_TOOLS = {"list_files": list_files, "read_file": read_file}

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


# --- NEW: the MCP component -------------------------------------------------

# which MCP server owns each remote tool (tool name -> live session)
mcp_sessions: dict[str, ClientSession] = {}


async def connect_mcp(stack: AsyncExitStack):
    """Launch each MCP server and merge its tools into TOOL_SCHEMAS."""
    for name, params in MCP_SERVERS.items():
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        tools = (await session.list_tools()).tools
        for tool in tools:
            mcp_sessions[tool.name] = session
            TOOL_SCHEMAS.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                    # MCP schemas may have optional fields; preserve that contract.
                    "strict": False,
                }
            )
        print(f"  [mcp] connected '{name}': {[t.name for t in tools]}")


async def call_tool(name: str, args: dict) -> str:
    """Run a tool - ours directly, or an MCP server's over the protocol."""
    if name in LOCAL_TOOLS:
        return LOCAL_TOOLS[name](**args)
    result = await mcp_sessions[name].call_tool(name, args)
    return "\n".join(c.text for c in result.content if hasattr(c, "text"))


# --- the agent loop, unchanged from v2 (just async now) ---------------------

async def run_agent(user_message: str) -> str:
    messages: ResponseInputParam = [
        {"role": "system", "content": "You are a helpful personal assistant."},
        {"role": "user", "content": user_message},
    ]
    while True:
        response = await client.responses.create(
            model=MODEL, input=messages, tools=TOOL_SCHEMAS
        )
        tool_calls = [item for item in response.output if item.type == "function_call"]
        if not tool_calls:
            return response.output_text

        # Replay reasoning and function calls together before the tool results.
        messages.extend(response.model_dump(exclude_unset=True)["output"])
        for call in tool_calls:
            args = json.loads(call.arguments or "{}")
            print(f"  [tool] {call.name}({args})")
            result = await call_tool(call.name, args)
            messages.append(
                {"type": "function_call_output", "call_id": call.call_id, "output": result}
            )


async def main():
    async with AsyncExitStack() as stack:
        stack.enter_context(credential)
        await stack.enter_async_context(client)
        await connect_mcp(stack)
        print(f"\nv3 assistant ({MODEL}) with {len(TOOL_SCHEMAS)} tools - ctrl+c to quit")
        try:
            while True:
                question = await asyncio.to_thread(input, "\nyou: ")
                print("\nassistant:", await run_agent(question))
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")


if __name__ == "__main__":
    asyncio.run(main())
