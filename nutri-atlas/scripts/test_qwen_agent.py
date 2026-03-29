"""
Test qwen-agent with tool calling using <tool_call> XML format.
Parses tool calls natively from the model output and executes them.
Make sure start_server.sh is running before executing this.
"""
import json
import re
from qwen_agent.llm import get_chat_model

# --- Tool registry ---
def get_weather(city: str) -> dict:
    """Fake weather tool for demo."""
    return {'city': city, 'temperature': '22°C', 'condition': 'Sunny'}

TOOLS = {
    'get_weather': get_weather,
}

TOOL_DEFINITIONS = [
    {
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get the current weather for a given city.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': {'type': 'string', 'description': 'The name of the city'},
                },
                'required': ['city'],
            },
        },
    }
]

# --- LLM config ---
llm = get_chat_model({
    'model': 'unsloth/Qwen3.5-9B-GGUF',
    'model_type': 'qwenvl_oai',
    'model_server': 'http://localhost:8001/v1',
    'api_key': 'EMPTY',
    'generate_cfg': {'thought_in_content': True},
})


def parse_tool_calls(text: str) -> list:
    """Extract all <tool_call>...</tool_call> blocks from model output."""
    calls = []
    for match in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
        try:
            calls.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            pass
    return calls


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def run_agent(user_query: str):
    print(f"User: {user_query}\n")
    messages = [{'role': 'user', 'content': user_query}]

    while True:
        # Collect the full streamed response
        full_content = ''
        for chunks in llm.chat(messages=messages, functions=TOOL_DEFINITIONS, stream=True):
            for chunk in chunks:
                if chunk.get('role') == 'assistant':
                    full_content = chunk.get('content', '')

        # Show thinking if present
        think_match = re.search(r'<think>(.*?)</think>', full_content, re.DOTALL)
        if think_match:
            print(f"[THINK] {think_match.group(1).strip()[:300]}...\n")

        clean_content = strip_think(full_content)

        # Check for tool calls
        tool_calls = parse_tool_calls(clean_content)
        if not tool_calls:
            # No tool calls — final answer
            print(f"[ANSWER]\n{clean_content}\n")
            break

        # Execute each tool call
        messages.append({'role': 'assistant', 'content': full_content})
        for call in tool_calls:
            name = call.get('name')
            args = call.get('arguments', call.get('parameters', {}))
            print(f"[TOOL CALL]  name = {name}")
            print(f"             args = {json.dumps(args)}")

            if name in TOOLS:
                result = TOOLS[name](**args)
                print(f"[TOOL RESULT] {json.dumps(result)}\n")
                messages.append({'role': 'function', 'content': json.dumps(result), 'name': name})
            else:
                print(f"[TOOL ERROR] Unknown tool: {name}\n")


if __name__ == '__main__':
    run_agent("What's the weather like in Tokyo and Paris?")
