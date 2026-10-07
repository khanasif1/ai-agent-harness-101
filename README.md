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
[v3_MCP.py](_src/v3_MCP.py), and [v4.py](_src/v4.py) call the `gpt-6-astra` deployment
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
- [v3_MCP.py](_src/v3_MCP.py): an async agent loop with local and MCP tools. Token
  acquisition runs in a worker thread so it does not block the event loop.
- [v4.py](_src/v4.py): local tools, persistent memory, and conversation history.

For Azure-hosted production use, prefer an explicitly configured
`ManagedIdentityCredential` over the local-development credential chain.

## v5: Model-switching harness with Entra authentication

[v5_harness.py](_src/v5_harness.py) uses the existing Foundry project
`https://use-ai-foundry-demo.services.ai.azure.com/api/projects/proj-default`.
It offers three deployments, with GPT selected by default:

| Command | Deployment | API |
| --- | --- | --- |
| `/model gpt` | `gpt-6-astra` | OpenAI Responses |
| `/model claude` | `claude-sonnet-5` | Anthropic Messages |
| `/model kimi` | `Kimi-K2.6` | OpenAI Chat Completions |

GPT and Kimi use the project endpoint with `/openai/v1/` appended. Claude uses
the resource endpoint `https://use-ai-foundry-demo.services.ai.azure.com/anthropic/`.
The project URL by itself is not an inference API URL. Deployment names must
match those in Foundry, including capitalization.

GPT uses Responses because this deployment does not support reasoning together
with function tools on Chat Completions. Tool-call history is translated between
the three APIs, including Claude-safe IDs for Kimi tool calls.

All three clients use `DefaultAzureCredential` and an automatically refreshed
token for `https://ai.azure.com/.default`. No OpenAI or Anthropic API keys are
required. Your signed-in identity needs project and model-inference access,
for example the **Azure AI User / Foundry User** role on the Foundry resource
and project. For production, use an explicit managed identity instead.

From `_src`, with the virtual environment activated:

```powershell
python -m pip install -r requirements.txt
az login
python .\v5_harness.py
```

Use `/models` to list deployments and `/model <alias>` to switch without
resetting the conversation. Local tools, MCP tools, and file-backed memory
remain available for every model. Provider-specific reasoning is retained for
the originating API but is not forwarded to a different provider. The local
Ollama entries and direct API-key endpoints are no longer part of v5.

See Microsoft's [Claude authentication guide](https://learn.microsoft.com/azure/foundry/foundry-models/how-to/use-foundry-models-claude)
and [Foundry reasoning-model client setup](https://learn.microsoft.com/azure/foundry/foundry-models/how-to/use-chat-reasoning).

## MCP package downloads

[v3_MCP.py](_src/v3_MCP.py) and [v5_harness.py](_src/v5_harness.py) launch `mcp-server-time`
and `mcp-server-fetch` through `uvx`, which is provided by the `uv` dependency.
`uvx` installs those server packages in separate environments on demand, so the
first run needs internet access.

If your network requires a package mirror, both examples support
`UV_DEFAULT_INDEX`. From `_src`, set it in the same PowerShell session before
starting the example, replacing the URL with your approved package index:

```powershell
$env:UV_DEFAULT_INDEX = "https://your-package-index.example/simple/"
python .\v3_MCP.py
```

Use `python .\v5_harness.py` instead to run v5. Each example passes the index
explicitly to the subprocess because the MCP stdio launcher does not inherit
`UV_DEFAULT_INDEX` automatically. Leaving it
unset or empty preserves the default `uvx` index behavior. If initialization
reports `Connection closed`, check the server's stderr above the traceback for
package download or startup errors.

## Offline tests

From `_src` with the virtual environment activated:

```powershell
python -m unittest discover -s tests -v
```

The [example regression tests](_src/tests/test_foundry_examples.py) and
[v5 harness tests](_src/tests/test_foundry_harness.py) use mocked HTTP and MCP
transports to check authentication, token refresh, function calls, cross-model
conversation history, and memory without Azure requests or changes to your
workspace files.


How to Build an Agentic Harness from Scratch - Full Python Tutorial - https://www.youtube.com/watch?v=H5o1P8RMiMw&t=2063s