"""
Qwen3.5-9B Assistant demo using local llama-server.
Mirrors the qwen-agent quickstart example format.

Make sure start_server.sh is running before executing this.
Usage:
    conda activate qwen
    python scripts/assistant_demo.py
"""
import json
import re
import json5
import urllib.parse
from qwen_agent.agents import Assistant
from qwen_agent.llm import get_chat_model
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.output_beautify import typewriter_print


# Step 1: Define custom tools using BaseTool.

@register_tool('get_weather')
class GetWeather(BaseTool):
    description = 'Get the current weather for a given city. Returns temperature and conditions.'
    parameters = [
        {
            'name': 'city',
            'type': 'string',
            'description': 'The name of the city to get weather for.',
            'required': True,
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        city = json5.loads(params)['city']
        # Fake data for demo — replace with a real weather API if needed.
        return json5.dumps(
            {'city': city, 'temperature': '22°C', 'condition': 'Sunny'},
            ensure_ascii=False,
        )


# Step 2: Configure the LLM (local llama-server).
llm_cfg = {
    'model': 'unsloth/Qwen3.5-9B-GGUF',
    'model_type': 'qwenvl_oai',
    'model_server': 'http://localhost:8001/v1',
    'api_key': 'EMPTY',
    'generate_cfg': {
        'thought_in_content': True,
    },
}

# Step 3: Create the Assistant agent.
system_instruction = (
    'You are a helpful assistant. '
    'When the user asks about weather, use the get_weather tool to fetch data. '
    'Always reply in the same language the user uses.'
)

bot = Assistant(
    llm=llm_cfg,
    system_message=system_instruction,
    function_list=['get_weather'],
)


def _parse_and_run_tool_calls(content: str) -> list[dict]:
    """
    Qwen3.5 small models output tool calls as <tool_call>JSON</tool_call>.
    qwen-agent's Assistant does not parse this format natively, so we handle
    it here and return tool result messages to feed back into the loop.
    """
    results = []
    for match in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', content, re.DOTALL):
        try:
            call = json.loads(match.group(1))
            name = call.get('name', '')
            args = call.get('arguments', call.get('parameters', {}))

            print(f'\n  [TOOL CALL]  → {name}({json.dumps(args)})')

            # Route to registered tools
            if name == 'get_weather':
                result = GetWeather().call(json.dumps(args))
            else:
                result = json.dumps({'error': f'Unknown tool: {name}'})

            print(f'  [TOOL RESULT] ← {result}')
            results.append({'role': 'function', 'name': name, 'content': result})
        except (json.JSONDecodeError, Exception) as e:
            print(f'  [TOOL ERROR] {e}')
    return results


# Step 4: Run as a chatbot with streaming output.
messages = []

print('Qwen3.5-9B Assistant (type "exit" to quit)\n')
while True:
    query = input('\nUser: ').strip()
    if query.lower() in ('exit', 'quit', ''):
        break

    messages.append({'role': 'user', 'content': query})

    response = []
    response_plain_text = ''
    print('Assistant:')

    for response in bot.run(messages=messages):
        response_plain_text = typewriter_print(response, response_plain_text)

    # Check last assistant message for tool calls and handle them
    for msg in response:
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            tool_results = _parse_and_run_tool_calls(content)
            if tool_results:
                # Feed tool results back and get final answer
                follow_up = list(response) + tool_results
                follow_up_plain = ''
                print('\nAssistant (after tools):')
                for follow_up_response in bot.run(messages=messages + follow_up):
                    follow_up_plain = typewriter_print(follow_up_response, follow_up_plain)
                response = follow_up_response
                break

    messages.extend(response)
