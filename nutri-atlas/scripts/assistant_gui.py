"""
Qwen3.5-9B Assistant — Gradio WebUI demo.
Launches a web interface for chatting with the assistant.

Make sure start_server.sh is running before executing this.
Usage:
    conda activate qwen
    python scripts/assistant_gui.py

Then open http://localhost:7860 in your browser.
"""
import json5
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.gui import WebUI


# Step 1: Define custom tools.

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
bot = Assistant(
    llm=llm_cfg,
    system_message=(
        'You are a helpful assistant. '
        'When the user asks about weather, use the get_weather tool. '
        'Always reply in the same language the user uses.'
    ),
    function_list=['get_weather'],
)

# Step 4: Launch the Gradio WebUI.
if __name__ == '__main__':
    WebUI(bot).run()
