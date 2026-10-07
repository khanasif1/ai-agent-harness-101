import importlib.util
import json
import os
import threading
import unittest
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx2 as httpx
from anthropic import AsyncAnthropicFoundry
from anthropic import PermissionDeniedError as AnthropicPermissionDeniedError
from azure.core.exceptions import ClientAuthenticationError
from openai import AsyncOpenAI, PermissionDeniedError

from test_foundry_examples import (
    FakeCredential,
    function_call as response_call,
    message as response_message,
    reasoning as response_reasoning,
    reply as response_reply,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "https://use-ai-foundry-demo.services.ai.azure.com/api/projects/proj-default"
SCOPE = "https://ai.azure.com/.default"


def function_call(name, arguments, call_id="call_test"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def chat_reply(text=None, calls=None, reasoning=None, finish_reason=None):
    message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = calls
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-deployment",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason or ("tool_calls" if calls else "stop"),
                    "message": message,
                }
            ],
        },
    )


def claude_reply(*blocks, stop_reason="end_turn"):
    return httpx.Response(
        200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": list(blocks),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )


class Harness:
    def __init__(self, replies):
        self.requests = []
        self.replies = iter(replies)
        self.credential = FakeCredential()
        transport = httpx.MockTransport(self.respond)

        class TestOpenAI(AsyncOpenAI):
            def __init__(self, **kwargs):
                super().__init__(
                    **kwargs,
                    http_client=httpx.AsyncClient(transport=transport),
                    max_retries=0,
                )

        class TestAnthropic(AsyncAnthropicFoundry):
            def __init__(self, **kwargs):
                super().__init__(
                    **kwargs,
                    http_client=httpx.AsyncClient(transport=transport),
                    max_retries=0,
                )

        spec = importlib.util.spec_from_file_location("test_harness", ROOT / "v5_harness.py")
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load v5_harness")
        self.module = importlib.util.module_from_spec(spec)
        with (
            patch("azure.identity.DefaultAzureCredential", return_value=self.credential),
            patch("openai.AsyncOpenAI", TestOpenAI),
            patch("anthropic.AsyncAnthropicFoundry", TestAnthropic),
            patch("pathlib.Path.is_file", return_value=False),
            patch("sys.stdout"),
        ):
            spec.loader.exec_module(self.module)

    def respond(self, request):
        self.requests.append(request)
        return next(self.replies)

    def body(self, index):
        return json.loads(self.requests[index].content)


class FoundryHarnessTests(unittest.IsolatedAsyncioTestCase):
    def harness(self, *replies):
        harness = Harness(replies)
        self.addCleanup(harness.credential.close)
        for client in harness.module.clients.values():
            self.addAsyncCleanup(client.close)
        return harness

    def assert_entra(self, harness):
        self.assertTrue(harness.requests)
        for index, request in enumerate(harness.requests):
            token_number = min(index + 1, 2)
            self.assertEqual(
                request.headers["authorization"], f"Bearer offline-test-token-{token_number}"
            )
            self.assertNotIn("api-key", request.headers)
            self.assertNotIn("x-api-key", request.headers)
            self.assertEqual(request.url.host, "use-ai-foundry-demo.services.ai.azure.com")
        self.assertEqual(
            harness.credential.scopes, [(SCOPE,)] * min(len(harness.requests), 2)
        )
        self.assertNotIn(threading.get_ident(), harness.credential.threads)

    def assert_closed(self, harness):
        self.assertTrue(harness.credential.closed)
        self.assertTrue(all(client.is_closed() for client in harness.module.clients.values()))

    def test_registry_has_only_requested_deployments(self):
        module = self.harness().module
        self.assertEqual(module.model_name, "gpt")
        self.assertEqual(
            {name: config.model for name, config in module.MODELS.items()},
            {"gpt": "gpt-6-astra", "claude": "claude-sonnet-5", "kimi": "Kimi-K2.6"},
        )
        for alias in ("gpt", "kimi"):
            self.assertEqual(module.MODELS[alias].base_url, f"{PROJECT}/openai/v1/")
        self.assertEqual(
            module.MODELS["claude"].base_url,
            "https://use-ai-foundry-demo.services.ai.azure.com/anthropic/",
        )

    async def test_all_models_use_entra_refresh_tokens_and_share_history(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "unused-test-key",
                "ANTHROPIC_API_KEY": "unused-test-key",
                "ANTHROPIC_FOUNDRY_API_KEY": "unused-test-key",
            },
        ):
            harness = self.harness(
                response_reply(response_message("GPT answer")),
                claude_reply({"type": "text", "text": "Claude answer"}),
                chat_reply("Kimi answer"),
            )
            module = harness.module
            self.assertEqual(await module.run_agent("First"), "GPT answer")
            self.assertTrue(module.handle_command("/model claude"))
            self.assertEqual(await module.run_agent("Second"), "Claude answer")
            self.assertTrue(module.handle_command("/model kimi"))
            self.assertEqual(await module.run_agent("Third"), "Kimi answer")

        self.assertEqual(
            [harness.body(i)["model"] for i in range(3)],
            ["gpt-6-astra", "claude-sonnet-5", "Kimi-K2.6"],
        )
        self.assertEqual(str(harness.requests[0].url), f"{PROJECT}/openai/v1/responses")
        self.assertEqual(
            [item["type"] for item in harness.body(0)["input"]], ["message", "message"]
        )
        self.assertEqual(
            str(harness.requests[1].url),
            "https://use-ai-foundry-demo.services.ai.azure.com/anthropic/v1/messages",
        )
        self.assertEqual(str(harness.requests[2].url), f"{PROJECT}/openai/v1/chat/completions")
        self.assertEqual(harness.body(1)["system"], module.messages[0].content)
        self.assertEqual(
            harness.body(1)["messages"][1],
            {"role": "assistant", "content": [{"type": "text", "text": "GPT answer"}]},
        )
        self.assertEqual(
            harness.body(2)["messages"][4],
            {"role": "assistant", "content": "Claude answer"},
        )
        self.assert_entra(harness)

    async def test_responses_reasoning_and_tools_replay_then_transfer_to_claude(self):
        calls = [
            response_reasoning("rs_test"),
            response_call("list_files", {}, "call_local"),
            response_call("get_time", {"timezone": "UTC"}, "call_remote"),
        ]
        harness = self.harness(
            response_reply(*calls),
            response_reply(response_message("Done")),
            claude_reply({"type": "text", "text": "Continued"}),
        )
        module = harness.module
        session = AsyncMock()
        session.call_tool.return_value = SimpleNamespace(
            content=[SimpleNamespace(text="12:00"), SimpleNamespace(text="UTC")]
        )
        with (
            patch.dict(module.LOCAL_TOOLS, list_files=Mock(return_value="notes.txt")),
            patch.dict(module.mcp_sessions, get_time=session),
        ):
            self.assertEqual(await module.run_agent("Files and time"), "Done")
            module.handle_command("/model claude")
            self.assertEqual(await module.run_agent("Continue"), "Continued")

        session.call_tool.assert_awaited_once_with("get_time", {"timezone": "UTC"})
        self.assertEqual(harness.body(1)["input"][2:5], calls)
        for tool in harness.body(0)["tools"]:
            self.assertNotIn("function", tool)
            self.assertIs(tool["strict"], False)
        self.assertEqual(
            harness.body(1)["input"][5:7],
            [
                {"type": "function_call_output", "call_id": "call_local", "output": "notes.txt"},
                {"type": "function_call_output", "call_id": "call_remote", "output": "12:00\nUTC"},
            ],
        )
        converted = harness.body(2)["messages"]
        self.assertEqual(
            converted[1]["content"],
            [
                {"type": "tool_use", "id": "toolu_2_0", "name": "list_files", "input": {}},
                {
                    "type": "tool_use",
                    "id": "toolu_2_1",
                    "name": "get_time",
                    "input": {"timezone": "UTC"},
                },
            ],
        )
        self.assertEqual(
            converted[2],
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_2_0", "content": "notes.txt"},
                    {"type": "tool_result", "tool_use_id": "toolu_2_1", "content": "12:00\nUTC"},
                ],
            },
        )
        self.assert_entra(harness)

    async def test_claude_tools_and_signed_thinking_replay_then_switch_to_gpt(self):
        blocks = [
            {"type": "thinking", "thinking": "Private test reasoning", "signature": "test-signature"},
            {"type": "text", "text": "Checking"},
            {"type": "tool_use", "id": "toolu_read", "name": "read_file", "input": {"filename": "notes.txt"}},
            {"type": "tool_use", "id": "toolu_save", "name": "save_memory", "input": {"fact": "test fact"}},
        ]
        harness = self.harness(
            claude_reply(*blocks, stop_reason="tool_use"),
            claude_reply({"type": "text", "text": "Done"}),
            response_reply(response_message("Continued")),
        )
        module = harness.module
        module.handle_command("/model claude")
        read = Mock(return_value="test note")
        save = Mock(return_value="saved")
        with patch.dict(module.LOCAL_TOOLS, read_file=read, save_memory=save):
            self.assertEqual(await module.run_agent("Read and remember"), "Done")
            module.handle_command("/model gpt")
            self.assertEqual(await module.run_agent("Continue"), "Continued")
        read.assert_called_once_with(filename="notes.txt")
        save.assert_called_once_with(fact="test fact")
        self.assertEqual(harness.body(1)["messages"][1]["content"], blocks)
        self.assertEqual(
            harness.body(1)["messages"][2]["content"],
            [
                {"type": "tool_result", "tool_use_id": "toolu_read", "content": "test note"},
                {"type": "tool_result", "tool_use_id": "toolu_save", "content": "saved"},
            ],
        )
        for tool in harness.body(0)["tools"]:
            self.assertNotIn("function", tool)
            self.assertEqual(tool["input_schema"]["type"], "object")
        self.assertEqual(harness.body(0)["max_tokens"], 16384)
        replay = harness.body(2)["input"][2]
        self.assertEqual(replay["type"], "message")
        self.assertEqual(replay["content"], "Checking")
        self.assertEqual(
            harness.body(2)["input"][3],
            {"type": "function_call", "call_id": "toolu_read", "name": "read_file", "arguments": '{"filename": "notes.txt"}'},
        )
        self.assertNotIn("Private test reasoning", json.dumps(harness.body(2)))
        self.assert_entra(harness)

    async def test_kimi_reasoning_survives_tool_rounds_and_is_not_sent_to_other_models(self):
        harness = self.harness(
            chat_reply(
                calls=[function_call("list_files", {}, "functions.list_files:0")],
                reasoning="Private Kimi reasoning",
            ),
            chat_reply("Done", reasoning="More private Kimi reasoning"),
            response_reply(response_message("GPT answer")),
            claude_reply({"type": "text", "text": "Claude answer"}),
            chat_reply("Resumed"),
        )
        module = harness.module
        module.handle_command("/model kimi")
        with patch.dict(module.LOCAL_TOOLS, list_files=Mock(return_value="notes.txt")):
            self.assertEqual(await module.run_agent("List files"), "Done")
        self.assertEqual(
            harness.body(1)["messages"][2]["reasoning_content"], "Private Kimi reasoning"
        )
        module.handle_command("/model gpt")
        await module.run_agent("Continue")
        module.handle_command("/model claude")
        await module.run_agent("Continue again")
        self.assertEqual(
            harness.body(3)["messages"][1]["content"][0]["id"], "toolu_2_0"
        )
        self.assertEqual(
            harness.body(3)["messages"][2]["content"][0]["tool_use_id"], "toolu_2_0"
        )
        for index in (2, 3):
            self.assertNotIn("Kimi reasoning", json.dumps(harness.body(index)))
        module.handle_command("/model kimi")
        self.assertEqual(await module.run_agent("Resume"), "Resumed")
        self.assertEqual(
            harness.body(4)["messages"][2]["reasoning_content"], "Private Kimi reasoning"
        )
        self.assert_entra(harness)

    async def test_mcp_preserves_optional_schema_and_forwards_package_index(self):
        schema = {
            "type": "object",
            "properties": {"timezone": {"type": "string"}, "format": {"type": "string"}},
            "required": ["timezone"],
        }
        for index in (None, "", "https://packages.example.test/simple/"):
            with self.subTest(index=index), patch.dict(os.environ), TemporaryDirectory() as directory:
                os.environ.pop("UV_DEFAULT_INDEX", None)
                if index is not None:
                    os.environ["UV_DEFAULT_INDEX"] = index
                harness = self.harness()
                module = harness.module
                config = Path(directory) / "mcp_servers.json"
                config.write_text(
                    json.dumps({"mcpServers": {"time": {"command": "uvx", "args": ["mcp-server-time"]}}}),
                    encoding="utf-8",
                )
                session = AsyncMock()
                session.__aenter__.return_value = session
                session.list_tools.return_value = SimpleNamespace(
                    tools=[SimpleNamespace(name="get_time", description=None, input_schema=schema)]
                )
                params = []

                @asynccontextmanager
                async def stdio(server):
                    params.append(server)
                    yield object(), object()

                with (
                    patch.object(module, "MCP_CONFIG", config),
                    patch.object(module, "stdio_client", stdio),
                    patch.object(module, "ClientSession", return_value=session),
                ):
                    async with AsyncExitStack() as stack:
                        await module.connect_mcp(stack)
                self.assertEqual(params[0].args, ["mcp-server-time"])
                self.assertEqual(params[0].env, {"UV_DEFAULT_INDEX": index} if index else None)
                self.assertEqual(module.TOOL_SCHEMAS[-1]["function"]["parameters"], schema)
                self.assertEqual(module.responses_tools()[-1]["parameters"], schema)
                self.assertIs(module.responses_tools()[-1]["strict"], False)
                self.assertEqual(module.claude_tools()[-1]["input_schema"], schema)
                self.assertEqual(module.claude_tools()[-1]["description"], "")
                module.handle_command("/tools")
                session.initialize.assert_awaited_once()
                session.__aexit__.assert_awaited_once()

    async def test_permission_errors_propagate_for_every_model(self):
        for alias in ("gpt", "claude", "kimi"):
            with self.subTest(model=alias):
                harness = self.harness(
                    httpx.Response(403, json={"error": {"type": "permission_error", "message": "Access denied"}})
                )
                harness.module.handle_command(f"/model {alias}")
                error = AnthropicPermissionDeniedError if alias == "claude" else PermissionDeniedError
                with self.assertRaises(error):
                    await harness.module.run_agent("Hello")
                self.assert_entra(harness)

    async def test_credential_errors_never_fall_back_to_api_keys(self):
        for alias in ("gpt", "claude", "kimi"):
            with self.subTest(model=alias):
                harness = self.harness()
                harness.credential.get_token = Mock(
                    side_effect=ClientAuthenticationError("Sign in with az login")
                )
                harness.module.handle_command(f"/model {alias}")
                with self.assertRaises(ClientAuthenticationError):
                    await harness.module.run_agent("Hello")
                self.assertFalse(harness.requests)

    async def test_empty_and_truncated_replies_fail_explicitly(self):
        cases = [
            ("gpt", response_reply(), "no text or tool calls"),
            ("kimi", chat_reply(reasoning="Only reasoning"), "no text or tool calls"),
            ("kimi", chat_reply("Partial", finish_reason="length"), "length"),
            ("kimi", chat_reply(finish_reason="content_filter"), "content_filter"),
            (
                "gpt",
                httpx.Response(
                    200,
                    json={
                        **response_reply().json(),
                        "status": "incomplete",
                        "incomplete_details": {"reason": "max_output_tokens"},
                    },
                ),
                "incomplete",
            ),
            ("claude", claude_reply(), "no text or tool calls"),
            ("claude", claude_reply(stop_reason="max_tokens"), "token limit"),
        ]
        for alias, reply, error in cases:
            with self.subTest(model=alias, error=error):
                harness = self.harness(reply)
                harness.module.handle_command(f"/model {alias}")
                with self.assertRaisesRegex(RuntimeError, error):
                    await harness.module.run_agent("Hello")

    async def test_invalid_tool_arguments_are_not_executed(self):
        for arguments in ("[", "[]"):
            with self.subTest(arguments=arguments):
                call = response_call("list_files", {}, "call_test")
                call["arguments"] = arguments
                harness = self.harness(response_reply(call))
                tool = Mock()
                with patch.dict(harness.module.LOCAL_TOOLS, list_files=tool):
                    with self.assertRaises(ValueError):
                        await harness.module.run_agent("List files")
                tool.assert_not_called()

    async def test_main_closes_clients_and_credentials_on_exit(self):
        for inputs in ([EOFError], ["/model claude", "/quit"]):
            with self.subTest(inputs=inputs):
                harness = self.harness()
                with (
                    patch.object(harness.module, "connect_mcp", AsyncMock()),
                    patch("builtins.input", side_effect=inputs),
                ):
                    await harness.module.main()
                self.assert_closed(harness)
                self.assertFalse(harness.requests)

    async def test_main_closes_clients_and_credentials_on_mcp_failure(self):
        harness = self.harness()
        with patch.object(
            harness.module, "connect_mcp", AsyncMock(side_effect=RuntimeError("MCP failed"))
        ):
            with self.assertRaisesRegex(RuntimeError, "MCP failed"):
                await harness.module.main()
        self.assert_closed(harness)

    def test_commands_reject_unknown_models_without_changing_selection(self):
        module = self.harness().module
        self.assertFalse(module.handle_command("Hello"))
        self.assertTrue(module.handle_command("/models"))
        self.assertTrue(module.handle_command("/model local"))
        self.assertEqual(module.model_name, "gpt")

    def test_claude_tool_ids_are_safe_and_distinct_across_repeated_kimi_calls(self):
        module = self.harness().module
        for _ in range(2):
            module.messages.extend(
                [
                    module.Message(role="user", content="List files"),
                    module.Message(
                        role="assistant",
                        tool_calls=[
                            module.ToolCall(
                                id="functions.list_files:0",
                                name="list_files",
                                arguments="{}",
                            )
                        ],
                    ),
                    module.Message(
                        role="tool", tool_call_id="functions.list_files:0", content="notes.txt"
                    ),
                ]
            )
        history = module.claude_messages()
        first_id = history[1]["content"][0]["id"]
        second_id = history[4]["content"][0]["id"]
        self.assertNotEqual(first_id, second_id)
        for call_id in (first_id, second_id):
            self.assertRegex(call_id, r"^[a-zA-Z0-9_-]+$")
        self.assertEqual(history[2]["content"][0]["tool_use_id"], first_id)
        self.assertEqual(history[5]["content"][0]["tool_use_id"], second_id)
        self.assertEqual(module.messages[2].tool_calls[0].id, "functions.list_files:0")

    def test_local_file_tools_and_memory_still_work(self):
        module = self.harness().module
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(module, "WORKSPACE", root),
                patch.object(module, "MEMORY_FILE", root / "memory.md"),
            ):
                self.assertEqual(module.load_memory(), "(nothing saved yet)")
                self.assertEqual(module.write_file("note.txt", "A note"), "wrote note.txt")
                self.assertEqual(module.read_file("note.txt"), "A note")
                self.assertIn("note.txt", module.list_files())
                self.assertEqual(module.save_memory("test fact"), "saved: test fact")
                self.assertEqual(module.load_memory(), "- test fact\n")


if __name__ == "__main__":
    unittest.main()
