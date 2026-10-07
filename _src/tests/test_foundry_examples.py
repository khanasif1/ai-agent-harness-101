import importlib.util
import json
import threading
import time
import unittest
from contextlib import AsyncExitStack, asynccontextmanager
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx2 as httpx
from azure.core.credentials import AccessToken
from openai import AsyncOpenAI, OpenAI, PermissionDeniedError

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://use-ai-foundry-demo.services.ai.azure.com/openai/v1/responses"
SCOPE = "https://cognitiveservices.azure.com/.default"


def message(text, item_id="msg_test"):
    return {
        "type": "message",
        "id": item_id,
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def reasoning(item_id):
    return {"type": "reasoning", "id": item_id, "summary": []}


def function_call(name, arguments, call_id):
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
        "status": "completed",
    }


def reply(*output):
    return httpx.Response(
        200,
        json={
            "id": "resp_test",
            "object": "response",
            "created_at": 0,
            "model": "gpt-6-astra",
            "status": "completed",
            "output": list(output),
        },
    )


def denied():
    return httpx.Response(
        403,
        json={"error": {"message": "Access denied", "code": "Forbidden"}},
    )


class FakeCredential:
    def __init__(self):
        self.scopes = []
        self.threads = []
        self.closed = False

    def get_token(self, *scopes, **kwargs):
        self.scopes.append(scopes)
        self.threads.append(threading.get_ident())
        number = len(self.scopes)
        lifetime = 60 if number == 1 else 3600
        return AccessToken(f"offline-test-token-{number}", int(time.time()) + lifetime)

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Example:
    def __init__(self, name, replies):
        self.requests = []
        self.credential = FakeCredential()
        self.replies = iter(replies)
        is_async = name == "v3"
        sdk_client = AsyncOpenAI if is_async else OpenAI
        http_client = httpx.AsyncClient if is_async else httpx.Client
        transport = httpx.MockTransport(self.respond)

        def create_client(**kwargs):
            return sdk_client(
                **kwargs, http_client=http_client(transport=transport), max_retries=0
            )

        spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {name}")
        self.module = importlib.util.module_from_spec(spec)
        constructor = "openai.AsyncOpenAI" if is_async else "openai.OpenAI"
        with (
            patch("azure.identity.DefaultAzureCredential", return_value=self.credential),
            patch(constructor, side_effect=create_client),
            patch("sys.stdout"),
        ):
            spec.loader.exec_module(self.module)

    def respond(self, request):
        self.requests.append(request)
        return next(self.replies)

    def body(self, index):
        return json.loads(self.requests[index].content)


class FoundryExampleTests(unittest.IsolatedAsyncioTestCase):
    def example(self, name, *replies):
        example = Example(name, replies)
        self.addCleanup(example.credential.close)
        if name == "v3":
            self.addAsyncCleanup(example.module.client.close)
        else:
            self.addCleanup(example.module.client.close)
        return example

    def assert_foundry_requests(self, example):
        self.assertTrue(example.requests)
        for index, request in enumerate(example.requests):
            self.assertEqual(request.method, "POST")
            self.assertEqual(str(request.url), ENDPOINT)
            token_number = min(index + 1, 2)
            self.assertEqual(
                request.headers["authorization"],
                f"Bearer offline-test-token-{token_number}",
            )
            self.assertNotIn("api-key", request.headers)
            body = example.body(index)
            self.assertEqual(body["model"], "gpt-6-astra")
            self.assertNotIn("messages", body)
            for tool in body.get("tools", []):
                self.assertEqual(tool["type"], "function")
                self.assertIn("name", tool)
                self.assertIn("parameters", tool)
                self.assertNotIn("function", tool)
                self.assertIs(tool["strict"], False)
        self.assertEqual(
            example.credential.scopes, [(SCOPE,)] * min(len(example.requests), 2)
        )

    def test_sync_direct_answers_reset_per_turn(self):
        for name, entrypoint in (("v1", "chat"), ("v2", "run_agent")):
            with self.subTest(example=name):
                example = self.example(name, reply(message("Hello")), reply(message("Hi")))
                run = getattr(example.module, entrypoint)
                self.assertEqual(run("First question"), "Hello")
                self.assertEqual(run("Second question"), "Hi")
                self.assertEqual(
                    example.body(1)["input"],
                    [
                        {"role": "system", "content": "You are a helpful personal assistant."},
                        {"role": "user", "content": "Second question"},
                    ],
                )
                self.assert_foundry_requests(example)

    def test_v1_executes_all_calls_in_one_round(self):
        output = [
            reasoning("rs_one_round"),
            function_call("list_files", {}, "call_list"),
            function_call("read_file", {"filename": "notes.txt"}, "call_read"),
        ]
        example = self.example("v1", reply(*output), reply(message("Done")))
        list_files = Mock(return_value="notes.txt")
        read_file = Mock(return_value="Notes")
        with patch.dict(example.module.TOOLS, list_files=list_files, read_file=read_file):
            self.assertEqual(example.module.chat("Read my notes"), "Done")

        list_files.assert_called_once_with()
        read_file.assert_called_once_with(filename="notes.txt")
        self.assertEqual(len(example.requests), 2)
        self.assertNotIn("tools", example.body(1))
        self.assertEqual(example.body(1)["input"][2:5], output)
        self.assertEqual(
            example.body(1)["input"][-2:],
            [
                {"type": "function_call_output", "call_id": "call_list", "output": "notes.txt"},
                {"type": "function_call_output", "call_id": "call_read", "output": "Notes"},
            ],
        )
        self.assert_foundry_requests(example)

    def test_v2_repeats_tools_and_preserves_reasoning(self):
        first_output = [
            reasoning("rs_read"),
            function_call("list_files", {}, "call_list"),
            function_call("read_file", {"filename": "notes.txt"}, "call_read"),
        ]
        second_output = [
            reasoning("rs_write"),
            function_call(
                "write_file",
                {"filename": "summary.txt", "content": "Summary"},
                "call_write",
            ),
        ]
        example = self.example(
            "v2", reply(*first_output), reply(*second_output), reply(message("Saved"))
        )
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "notes.txt").write_text("Notes", encoding="utf-8")
            with patch.object(example.module, "WORKSPACE", workspace):
                self.assertEqual(example.module.run_agent("Summarize my notes"), "Saved")
            self.assertEqual((workspace / "summary.txt").read_text(), "Summary")

        self.assertEqual(len(example.requests), 3)
        self.assertEqual(example.body(2)["input"][2:5], first_output)
        self.assertEqual(example.body(2)["input"][-3:-1], second_output)
        self.assertEqual(
            example.body(2)["input"][-1],
            {"type": "function_call_output", "call_id": "call_write", "output": "wrote summary.txt"},
        )
        self.assertTrue(all(example.body(i)["tools"] for i in range(3)))
        self.assert_foundry_requests(example)

    def test_v4_preserves_memory_and_history_across_turns(self):
        first_output = [
            reasoning("rs_memory"),
            function_call("save_memory", {"fact": "Likes Python"}, "call_memory"),
        ]
        final_output = [reasoning("rs_answer"), message("Remembered", "msg_saved")]
        example = self.example(
            "v4", reply(*first_output), reply(*final_output), reply(message("Python"))
        )
        module = example.module
        with TemporaryDirectory() as directory:
            memory_file = Path(directory) / "memory.md"
            memory_file.write_text("- Lives in Melbourne\n", encoding="utf-8")
            with patch.object(module, "MEMORY_FILE", memory_file):
                messages = [
                    {"role": "system", "content": module.SYSTEM_PROMPT.format(memory=module.load_memory())},
                    {"role": "user", "content": "I like Python"},
                ]
                self.assertEqual(module.run_agent(messages), "Remembered")
                self.assertEqual(module.load_memory(), "- Lives in Melbourne\n- Likes Python\n")
                self.assertEqual(messages[-2:], final_output)
                messages.append({"role": "user", "content": "What language do I like?"})
                expected_history = deepcopy(messages)
                self.assertEqual(module.run_agent(messages), "Python")

        self.assertEqual(example.body(2)["input"], expected_history)
        self.assertEqual(messages[-1]["content"][0]["text"], "Python")
        self.assertEqual(example.body(1)["input"][2:4], first_output)
        self.assert_foundry_requests(example)

    def test_sync_permission_errors_propagate(self):
        for name in ("v1", "v2", "v4"):
            with self.subTest(example=name):
                example = self.example(name, denied())
                with self.assertRaises(PermissionDeniedError):
                    if name == "v1":
                        example.module.chat("Hello")
                    elif name == "v2":
                        example.module.run_agent("Hello")
                    else:
                        example.module.run_agent([{"role": "user", "content": "Hello"}])
                self.assert_foundry_requests(example)

    def test_invalid_tool_arguments_are_not_hidden(self):
        output = function_call("read_file", {}, "call_invalid")
        output["arguments"] = "{invalid"
        example = self.example("v2", reply(output))
        with self.assertRaises(json.JSONDecodeError):
            example.module.run_agent("Read my notes")
        self.assertEqual(len(example.requests), 1)

    async def test_v3_direct_answers_and_nonblocking_token_refresh(self):
        example = self.example("v3", reply(message("Hello")), reply(message("Hi")))
        self.assertEqual(await example.module.run_agent("First question"), "Hello")
        self.assertEqual(await example.module.run_agent("Second question"), "Hi")
        self.assertEqual(len(example.body(1)["input"]), 2)
        self.assertEqual(example.body(1)["input"][-1]["content"], "Second question")
        self.assertTrue(example.credential.threads)
        self.assertNotIn(threading.get_ident(), example.credential.threads)
        self.assert_foundry_requests(example)

    async def test_v3_connects_and_calls_local_and_mcp_tools(self):
        output = [
            reasoning("rs_mcp"),
            function_call("list_files", {}, "call_local"),
            function_call("get_time", {"timezone": "Australia/Sydney"}, "call_mcp"),
        ]
        example = self.example("v3", reply(*output), reply(message("Done")))
        module = example.module
        schema = {
            "type": "object",
            "properties": {
                "timezone": {"type": "string"},
                "format": {"type": "string"},
            },
            "required": ["timezone"],
        }
        session = AsyncMock()
        session.__aenter__.return_value = session
        session.list_tools.return_value = SimpleNamespace(
            tools=[SimpleNamespace(name="get_time", description=None, input_schema=schema)]
        )
        session.call_tool.return_value = SimpleNamespace(
            content=[SimpleNamespace(text="16:00"), SimpleNamespace(text="Sydney")]
        )

        @asynccontextmanager
        async def stdio(_):
            yield object(), object()

        with (
            patch.object(module, "MCP_SERVERS", {"time": module.MCP_SERVERS["time"]}),
            patch.object(module, "stdio_client", stdio),
            patch.object(module, "ClientSession", return_value=session),
            patch.dict(module.LOCAL_TOOLS, list_files=Mock(return_value="notes.txt")),
        ):
            async with AsyncExitStack() as stack:
                await module.connect_mcp(stack)
                self.assertEqual(await module.run_agent("List files and get the time"), "Done")

        session.initialize.assert_awaited_once()
        session.call_tool.assert_awaited_once_with("get_time", {"timezone": "Australia/Sydney"})
        session.__aexit__.assert_awaited_once()
        tool = next(tool for tool in example.body(0)["tools"] if tool["name"] == "get_time")
        self.assertEqual(tool["parameters"], schema)
        self.assertIs(tool["strict"], False)
        self.assertEqual(example.body(1)["input"][2:5], output)
        self.assertEqual(
            example.body(1)["input"][-2:],
            [
                {"type": "function_call_output", "call_id": "call_local", "output": "notes.txt"},
                {"type": "function_call_output", "call_id": "call_mcp", "output": "16:00\nSydney"},
            ],
        )
        self.assert_foundry_requests(example)

    async def test_v3_permission_errors_propagate(self):
        example = self.example("v3", denied())
        with self.assertRaises(PermissionDeniedError):
            await example.module.run_agent("Hello")
        self.assert_foundry_requests(example)

    async def test_v3_closes_resources_on_eof(self):
        example = self.example("v3")
        with (
            patch.object(example.module, "connect_mcp", AsyncMock()),
            patch("builtins.input", side_effect=EOFError),
        ):
            await example.module.main()
        self.assertTrue(example.module.client.is_closed())
        self.assertTrue(example.credential.closed)
        self.assertFalse(example.requests)

    async def test_v3_closes_resources_on_mcp_failure(self):
        example = self.example("v3")
        with patch.object(
            example.module, "connect_mcp", AsyncMock(side_effect=RuntimeError("MCP failed"))
        ):
            with self.assertRaisesRegex(RuntimeError, "MCP failed"):
                await example.module.main()
        self.assertTrue(example.module.client.is_closed())
        self.assertTrue(example.credential.closed)


if __name__ == "__main__":
    unittest.main()
