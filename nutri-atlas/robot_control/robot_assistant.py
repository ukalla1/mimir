"""
Robot navigation assistant powered by Qwen Agent.

The agent can:
  - list_landmarks       : show all named positions in the map
  - navigate_to_landmark : send the robot to a named position and report result

Setup:
    export ROBOT_IP=<robot onboard IP>    # default: 127.0.0.1
    export ROBOT_PORT=5555                # default: 5555
    export NAV_TIMEOUT_MS=60000          # default: 60000 (60s)

    conda activate qwen
    python robot_control/robot_assistant.py

Make sure start_server.sh is running first.
"""
import json
import re

from qwen_agent.llm import get_chat_model

from tools.navigate_tool import NavigateToLandmark, ListLandmarks  # noqa: F401
from tools.object_tool import GetDetectedObjects  # noqa: F401
from tools.motion_tool import SpinRobot, MoveRobot  # noqa: F401
from tools.object_tool import GetCurrentDetectedObjects  # noqa: F401
from tools.lidar_tool import GetLidarScan  # noqa: F401
from tools.nutrition_tool import GetMealRecommendation  # noqa: F401


# --- LLM config ---
llm = get_chat_model({
    'model': 'unsloth/Qwen3.5-9B-GGUF',
    'model_type': 'qwenvl_oai',
    'model_server': 'http://localhost:8080/v1',
    'api_key': 'EMPTY',
    'generate_cfg': {'thought_in_content': True},
})

# --- Tool definitions (OpenAI function format) ---
TOOL_DEFINITIONS = [
    {
        'type': 'function',
        'function': {
            'name': 'list_landmarks',
            'description': 'List all named landmarks the robot can navigate to, including their positions and descriptions.',
            'parameters': {
                'type': 'object',
                'properties': {},
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_detected_objects',
            'description': (
                'Return the full map of objects currently detected by the robot camera. '
                'Each entry contains the object name and its estimated position. '
                'The list updates in real time, but only while the robot is moving — '
                'if the list is not changing, the robot has either stopped or has fully explored the environment. '
                'Call this periodically after each movement to check whether a target object has appeared.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {},
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_current_detected_objects',
            'description': (
                'Return only the objects the robot camera is detecting RIGHT NOW, '
                'filtered by how recently they were last seen. '
                'Unlike get_detected_objects, this excludes stale entries from earlier exploration. '
                'Use this to confirm what is currently visible before making a navigation decision.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {},
            },
        },
    },
    # {
    #     'type': 'function',
    #     'function': {
    #         'name': 'get_lidar_scan',
    #         'description': (
    #             'Return the current LiDAR obstacle distances in 8 directional sectors around the robot. '
    #             'Returns min distance per sector, overall min_distance, its direction, and safe_directions. '
    #             'Use this before moving to check for nearby obstacles and choose a safe direction.'
    #         ),
    #         'parameters': {
    #             'type': 'object',
    #             'properties': {
    #                 'min_dist': {
    #                     'type': 'number',
    #                     'description': 'Minimum clearance in metres to consider a direction safe. Default 1.0 m.',
    #                 },
    #             },
    #         },
    #     },
    # },
    {
        'type': 'function',
        'function': {
            'name': 'spin_robot',
            'description': (
                'Rotate the robot in place by a given angle without moving forward or backward. '
                'Positive angle = counter-clockwise (left turn); negative = clockwise (right turn).'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'angle_deg': {
                        'type': 'number',
                        'description': 'Angle to rotate in degrees. Positive = CCW (left), negative = CW (right).',
                    },
                },
                'required': ['angle_deg'],
            },
        },
    },
    # {
    #     'type': 'function',
    #     'function': {
    #         'name': 'move_robot',
    #         'description': (
    #             'Move the robot straight forward or backward by a given distance in metres. '
    #             'Positive = forward, negative = backward.'
    #         ),
    #         'parameters': {
    #             'type': 'object',
    #             'properties': {
    #                 'distance_m': {
    #                     'type': 'number',
    #                     'description': 'Distance in metres. Positive = forward, negative = backward.',
    #                 },
    #             },
    #             'required': ['distance_m'],
    #         },
    #     },
    # },
    {
        'type': 'function',
        'function': {
            'name': 'get_meal_recommendation',
            'description': (
                'Get a personalized meal recommendation based on what the user has eaten. '
                'Uses nutritional gap analysis and a food knowledge base to suggest what to eat next. '
                'Example: user ate "an apple and a cup of milk" for breakfast, '
                'this tool recommends what to have for lunch.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'eaten_foods': {
                        'type': 'string',
                        'description': 'Description of what the user has eaten, e.g. "an apple and a cup of milk".',
                    },
                    'eaten_meal_type': {
                        'type': 'string',
                        'description': 'Which meal the eaten food was for: "breakfast", "lunch", "dinner", or "snack". Default: "breakfast".',
                    },
                    'next_meal': {
                        'type': 'string',
                        'description': 'Which meal to recommend for: "breakfast", "lunch", "dinner", or "snack". Default: "lunch".',
                    },
                },
                'required': ['eaten_foods'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'navigate_to_landmark',
            'description': (
                'Navigate the robot to a position in the map. Two use cases: '
                '(1) Named landmark — provide landmark_name and the position is looked up automatically; '
                'use list_landmarks first if unsure of available names. '
                '(2) Detected object — call get_detected_objects first to find the object\'s position (px, py), '
                'then call this tool with x and y directly, omitting landmark_name.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'landmark_name': {
                        'type': 'string',
                        'description': 'The name of the destination landmark (e.g. "kitchen", "lab"). Omit when navigating by coordinates.',
                    },
                    'x': {
                        'type': 'number',
                        'description': 'Map x coordinate to navigate to. Use when navigating to a detected object position.',
                    },
                    'y': {
                        'type': 'number',
                        'description': 'Map y coordinate to navigate to. Use when navigating to a detected object position.',
                    },
                },
            },
        },
    },
]

# --- Tool instances ---
_TOOLS = {
    'list_landmarks':        ListLandmarks(),
    'navigate_to_landmark':  NavigateToLandmark(),
    'get_detected_objects':  GetDetectedObjects(),
    'spin_robot':                    SpinRobot(),
    # 'move_robot':                    MoveRobot(),
    'get_current_detected_objects':  GetCurrentDetectedObjects(),
    # 'get_lidar_scan':                GetLidarScan(),
    'get_meal_recommendation':       GetMealRecommendation(),
}

SYSTEM_MSG = {
    'role': 'system',
    'content': (
        'You are a robot assistant that can navigate and provide nutritional advice. You have these tools:\n\n'
        'Navigation:\n'
        '- list_landmarks: list all navigable locations.\n'
        '- navigate_to_landmark: go to a named landmark (landmark_name), or directly to coordinates (x, y) from a detected object.\n'
        '- spin_robot: rotate in place in degrees (positive=CCW, negative=CW).\n'
        '- get_detected_objects: persistent map of all ever-seen objects with their positions. Updates only while the robot moves.\n'
        '- get_current_detected_objects: objects visible RIGHT NOW (no stale entries). Use after each spin.\n\n'
        'Nutrition:\n'
        '- get_meal_recommendation: given what the user has eaten, recommend what to eat next based on nutritional gap analysis.\n\n'
        'To explore a location, you MUST do a full 360° scan: '
        'spin 120° then get_current_detected_objects — repeat 3 times.\n\n'
        'To find an object:\n'
        '1. Call get_detected_objects — if found, navigate to its coordinates directly.\n'
        '2. If not, do the 3-spin scan at the current location.\n'
        '3. If still not found, navigate to the next landmark and repeat from step 2.\n'
        '4. If all landmarks exhausted, report the object as not found.\n\n'
        'For nutrition questions (e.g. "I ate X, what should I eat for lunch?"), '
        'use get_meal_recommendation with the foods the user described.\n\n'
        'Always reply in the same language the user uses.'
        'When you moving between living room and bedroom, you have to go hallway first.'
    ),
}


def _strip_think(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def _parse_tool_calls(text: str) -> list:
    calls = []
    for match in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
        try:
            calls.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            pass
    return calls


def _run_turn(messages: list) -> str:
    """
    Run one LLM turn (with tool loop). Returns the final assistant text reply.
    Modifies messages in-place by appending assistant + tool result messages.
    """
    while True:
        # Collect full streamed response
        full_content = ''
        for chunks in llm.chat(messages=messages, functions=TOOL_DEFINITIONS, stream=True):
            for chunk in chunks:
                if chunk.get('role') == 'assistant':
                    full_content = chunk.get('content', '')

        clean = _strip_think(full_content)
        messages.append({'role': 'assistant', 'content': clean or ' '})
        tool_calls = _parse_tool_calls(clean)

        if not tool_calls:
            if not clean.strip():
                # LLM emitted only <think> content mid-task — nudge it to continue
                messages.append({'role': 'user', 'content': 'Continue.'})
                continue
            return clean   # final answer — no more tool calls

        # Execute each tool call and append results
        for call in tool_calls:
            name = call.get('name', '')
            args = call.get('arguments', call.get('parameters', {}))
            # print(f'  [TOOL CALL]  → {name}({json.dumps(args)})')

            tool = _TOOLS.get(name)
            result = tool.call(json.dumps(args)) if tool else json.dumps({'error': f'Unknown tool: {name}'})
            # print(f'  [TOOL RESULT] ← {result}\n')

            messages.append({'role': 'function', 'name': name, 'content': result})


# --- Chat loop ---
def main():
    import os
    print('Robot Navigation Assistant (type "exit" to quit)')
    print(f'  Robot IP  : {os.environ.get("ROBOT_IP", "127.0.0.1")}')
    print(f'  Robot Port: {os.environ.get("ROBOT_PORT", "5555")}')
    print()

    messages = [SYSTEM_MSG]

    while True:
        query = input('User: ').strip()
        if query.lower() in ('exit', 'quit', ''):
            break

        messages.append({'role': 'user', 'content': query})

        print('Assistant: ', end='', flush=True)
        reply = _run_turn(messages)
        print(reply)
        print()


if __name__ == '__main__':
    main()
