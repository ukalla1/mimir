"""
Test client for Qwen3.5-9B llama-server.
Run after starting start_server.sh.
"""
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="EMPTY",
)

MODEL = "unsloth/Qwen3.5-9B-GGUF"

def test_thinking_mode():
    print("=== Thinking Mode ===")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is 15 * 27? Show your reasoning."}],
        max_tokens=4096,
        temperature=0.6,
        top_p=0.95,
    )
    msg = response.choices[0].message
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        print(f"<think>\n{msg.reasoning_content}\n</think>")
    print(f"Answer: {msg.content}")

def test_general():
    print("\n=== General Task ===")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=256,
        temperature=0.7,
        top_p=0.8,
    )
    print(f"Answer: {response.choices[0].message.content}")

if __name__ == "__main__":
    test_thinking_mode()
    test_general()
