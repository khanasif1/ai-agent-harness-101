# agent-harness-101

Progressive Python examples for building an agent harness.

## Setup

Requires Python 3.13 or newer. From the repository root, run in PowerShell:

```powershell
cd .\_src
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The runtime dependencies are listed in [requirements.txt](_src/requirements.txt)
and [pyproject.toml](_src/pyproject.toml). Standard-library modules do not need
separate installation.

## v0-v4: Microsoft Foundry with Entra authentication

[v0.py](_src/v0.py), [v1.py](_src/v1.py), [v2.py](_src/v2.py),
[v3.py](_src/v3.py), and [v4.py](_src/v4.py) call the `gpt-6-astra` deployment
using the Responses API at
`https://use-ai-foundry-demo.services.ai.azure.com/openai/v1/responses`.
They use `DefaultAzureCredential` with an automatically refreshed Microsoft Entra
token; no API key is required.

Install the [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli)
and ensure your signed-in identity has the **Cognitive Services OpenAI User**
role on the Foundry resource. Then, from the activated environment above:

```powershell
az login
python v0.py
```

Replace `v0.py` with the example you want to run:

- [v1.py](_src/v1.py): one round of local tool calls.
- [v2.py](_src/v2.py): repeated local tool calls until the model answers.
- [v3.py](_src/v3.py): an async agent loop with local and MCP tools. Token
  acquisition runs in a worker thread so it does not block the event loop.
- [v4.py](_src/v4.py): local tools, persistent memory, and conversation history.

For Azure-hosted production use, prefer an explicitly configured
`ManagedIdentityCredential` over the local-development credential chain.

## Model-switching harness

[harness.py](_src/harness.py) still defaults to
[Ollama](https://ollama.com/) at `http://localhost:11434/v1` with
`qwen3.5:4b`. Install and start Ollama, then run:

```powershell
ollama pull qwen3.5:4b
python harness.py
```

[v3.py](_src/v3.py) and [harness.py](_src/harness.py) launch `mcp-server-time`
and `mcp-server-fetch` through `uvx`, which is provided by the `uv` dependency.
`uvx` installs those server packages in separate environments on demand, so the
first run needs internet access.

## Offline tests

From `_src` with the virtual environment activated:

```powershell
python -m unittest discover -s tests -v
```

The [regression tests](_src/tests/test_foundry_examples.py) use mocked HTTP and
MCP transports to check authentication, function calls, conversation history,
and memory without Azure requests or changes to your workspace files.


How to Build an Agentic Harness from Scratch - Full Python Tutorial - https://www.youtube.com/watch?v=H5o1P8RMiMw&t=2063s