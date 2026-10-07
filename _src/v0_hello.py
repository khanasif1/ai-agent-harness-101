import sys

sys.stdout.reconfigure(encoding="utf-8")  # so emoji don't crash the Windows console

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)
client = OpenAI(
    # The SDK appends /responses to this base URL.
    base_url="https://use-ai-foundry-demo.services.ai.azure.com/openai/v1/",
    api_key=token_provider,
)
MODEL = "gpt-6-astra"


def chat(user_message: str) -> str:
    response = client.responses.create(
        model=MODEL,
        instructions="You are a helpful personal assistant.",
        input=user_message,
    )
    return response.output_text


if __name__ == "__main__":
    print(f"v0 assistant ({MODEL}) - ctrl+c to quit")
    try:
        while True:
            question = input("\nyou: ")
            print("\nassistant:", chat(question))
    except (EOFError, KeyboardInterrupt):
        print("\nbye!")

