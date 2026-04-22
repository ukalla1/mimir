"""
Robot navigation assistant powered by Qwen Agent.

The agent can:
  - list_landmarks       : show all named positions in the map
  - navigate_to_landmark : send the robot to a named position and report result

Setup (env vars or CLI args):
    python robot_control/robot_assistant.py --robot-ip 192.168.0.114 --robot-port 5555

    # or via env vars:
    export ROBOT_IP=192.168.0.114
    export ROBOT_PORT=5555
    export NAV_TIMEOUT_MS=60000

    conda activate qwen
    python robot_control/robot_assistant.py

Make sure start_server.sh is running first.
"""
import argparse
import base64
import json
import mimetypes
import os
import re

# --- Tool definitions (OpenAI function format) ---
TOOL_DEFINITIONS = [
    {
        'type': 'function',
        'function': {
            'name': 'list_landmarks',
            'description': 'List all locations the robot can navigate to. Includes both fixed named landmarks AND any objects recently detected by the camera. Always call this tool fresh — detected objects change every time the camera scans.',
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
                'Return the persistent landmark history stored on the robot. '
                'Each entry contains the object name and its map-frame position. '
                'This only changes after register_objects is called. '
                'Call this to check whether a target object has been previously registered.'
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
            'name': 'forget_object',
            'description': 'Remove a detected object from the persistent map by its frame name (e.g. "detected_bottle_0"). Use when an object is stale, was detected incorrectly, or has moved.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'frame_name': {
                        'type': 'string',
                        'description': 'Exact frame name to remove, e.g. "detected_bottle_0".',
                    },
                },
                'required': ['frame_name'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'scan_objects',
            'description': (
                'Run YOLO object detection on the current camera frame. '
                'Detects objects and stores them in TEMP MEMORY only — does NOT persist to landmark history. '
                'Use this to look at what is around without side effects. '
                'Call register_objects afterwards to persist detected objects as landmarks.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'targets': {
                        'type': 'string',
                        'description': 'Comma-separated labels to look for (e.g. "bottle,cup"). Empty = detect all.',
                    },
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'register_objects',
            'description': (
                'Persist detected objects as permanent landmarks on the robot. '
                'If navigate_and_scan just accumulated detections, this call drains that temp '
                'memory (map-frame positions, deduplicated against existing landmarks). '
                'Otherwise it runs a fresh 1-frame scan at the current camera pose. '
                'Only call this when the user asks to remember/save objects, or when actively '
                'searching for a specific object to navigate to.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'targets': {
                        'type': 'string',
                        'description': 'Comma-separated labels to register (e.g. "bottle,cup"). Empty = use default interest list or register all.',
                    },
                },
            },
        },
    },
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
            'name': 'navigate_and_scan',
            'description': (
                'Navigate to a destination WHILE running YOLO detection along the way. '
                'USE THIS (not navigate_to_landmark) whenever the user combines navigation '
                'with any observation intent, e.g. "go to X and look for objects", '
                '"navigate to X and detect", "on the way / along the way / in your way '
                'find/look for/scan/detect/remember anything", "as you move observe", '
                '"go to X and see what\'s there". '
                'Detections are stored in temp memory with map-frame positions. After arrival, '
                'call register_objects to persist them.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'landmark_name': {
                        'type': 'string',
                        'description': 'Destination landmark name (e.g. "kitchen", "hallway").',
                    },
                    'x': {
                        'type': 'number',
                        'description': 'Map x coordinate. Use when navigating by coordinates.',
                    },
                    'y': {
                        'type': 'number',
                        'description': 'Map y coordinate. Use when navigating by coordinates.',
                    },
                    'targets': {
                        'type': 'string',
                        'description': 'Comma-separated labels to detect (e.g. "bottle,cup"). Empty = detect all.',
                    },
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'navigate_to_landmark',
            'description': (
                'Navigate the robot to a position in the map — PURE navigation, no observation. '
                'DO NOT use this tool if the user mentions looking, scanning, detecting, checking, '
                'observing, or remembering objects during the trip — use navigate_and_scan instead. '
                'Use this tool ONLY when the user just wants to travel to a place. '
                'Two inputs: (1) landmark_name (looked up via list_landmarks); '
                '(2) x, y coordinates directly (for a previously detected object — '
                'call get_detected_objects first to find (px, py)).'
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


SYSTEM_MSG = {
    'role': 'system',
    'content': (
        'You are a robot assistant that can navigate and provide nutritional advice.\n\n'
        'Tool selection for any movement + observation request:\n'
        '  Just travel, no observation mentioned           → navigate_to_landmark\n'
        '  Observation at current location, no travel      → scan_objects\n'
        '  Travel AND observation during the trip          → navigate_and_scan\n'
        '  Travel AND save objects seen on the trip        → navigate_and_scan,\n'
        '                                                    then register_objects\n'
        '  Observe in all directions without moving away   → spin_robot + scan_objects (×4)\n\n'
        'Examples (follow these patterns exactly):\n'
        '  "Go to the kitchen"\n'
        '      → navigate_to_landmark(landmark_name="kitchen")\n'
        '  "What do you see?" / "Inspect this area"\n'
        '      → scan_objects\n'
        '  "Go to reception and look for objects on the way"\n'
        '      → navigate_and_scan(landmark_name="reception")\n'
        '  "Navigate to the lab and detect anything interesting"\n'
        '      → navigate_and_scan(landmark_name="lab")\n'
        '  "Go back to the start point, detect objects along the way"\n'
        '      → navigate_and_scan(landmark_name="start")\n'
        '  "Go to kitchen and remember anything you find"\n'
        '      → navigate_and_scan(landmark_name="kitchen")\n'
        '      → register_objects\n'
        '  "Rotate 90 degrees" / "turn left"\n'
        '      → spin_robot\n\n'
        'Tool reference (use the decision table above to pick; consult these for arguments):\n'
        '- list_landmarks: list ALL navigable locations — fixed landmarks AND recently detected objects. Always call fresh.\n'
        '- get_detected_objects: persistent landmark history on the robot. Only changes after register_objects. Call fresh.\n'
        '- scan_objects: YOLO on the current camera frame → TEMP MEMORY only. Non-destructive.\n'
        '- register_objects: persist detections as landmarks. Auto-drains temp memory from navigate_and_scan (map-frame); else runs a fresh 1-frame scan at the current pose.\n'
        '- navigate_and_scan: travel + detect along the way (see decision table).\n'
        '- navigate_to_landmark: PURE travel, no observation (see decision table).\n'
        '- spin_robot: rotate in place (degrees, +CCW / -CW).\n'
        '- forget_object: remove a stale detected object from the persistent map by frame name.\n'
        '- get_meal_recommendation: recommend food based on nutritional gap from what the user has eaten.\n\n'
        'IMPORTANT:\n'
        '- scan_objects never stores anything to landmark history.\n'
        '- register_objects is the ONLY way to add detected objects to landmark history — only call when the user asks to remember/save, or when actively searching for a specific object.\n'
        '- Never navigate to an object unless the user explicitly asks you to go there.\n'
        '- **Navigation failure handling**: if a navigation call you just issued returns status=failed or status=timeout, STOP for this turn. Do not try other landmarks, do not retry the same goal. Report the exact failure message and wait for the user. This applies ONLY to a failure the tool returned in the CURRENT user request — not to failures remembered from earlier turns.\n'
        '- **Each new user request is a clean slate**: a past nav failure does NOT block future requests. If the user asks again, you MUST attempt the tool call fresh — never refuse based on memory of a prior failure, never fabricate a tool reply.\n'
        '- **Never hallucinate a tool result**: only say a tool "failed" / "returned X" if you actually just called it in this turn and saw that reply. If you have not called the tool yet in the current turn, call it.\n'
        '- Trust the nav tool\'s reply. If it reports success, the robot arrived. If it reports failed (including "not actually arrived"), the robot did NOT arrive — tell the user honestly, do not claim success.\n\n'
        'To SEARCH for a specific object (only when the user says "find X" or "search for X" — '
        'NOT plain "go to X", which is a direct navigate, not a search):\n'
        '1. Call get_detected_objects — if found, navigate to its coordinates.\n'
        '2. If not, call register_objects(targets="X") at current location — scans, validates, stores.\n'
        '3. If still not found, do a 360° explore: spin 90° + register_objects(targets="X"), repeat 4 times.\n'
        '4. If still not found, report "not found nearby" and ask the user where to look. Do NOT auto-travel to other landmarks.\n\n'
        'For nutrition questions (e.g. "I ate X, what should I eat for lunch?"), '
        'use get_meal_recommendation with the foods the user described.\n\n'
        'Always reply in the same language the user uses. '
        'When moving between living room and bedroom, you have to go hallway first.\n\n'
        'If the user message contains an image, describe or analyse that image directly. '
        'Do NOT call scan_objects or any other tool for user-provided images.'
    ),
}


def _build_user_content(query: str):
    """Return content for a user message — plain string or multimodal list.

    Syntax: @/path/to/image.jpg optional text describing what to ask
    Example: @/tmp/photo.jpg what objects do you see?
    """
    if query.startswith('@'):
        parts = query[1:].split(' ', 1)
        img_path = os.path.expanduser(parts[0].strip())
        text = parts[1].strip() if len(parts) > 1 else 'What do you see?'
        from qwen_agent.llm.schema import ContentItem
        mime = mimetypes.guess_type(img_path)[0] or 'image/jpeg'
        with open(img_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        return [
            ContentItem(image=f'data:{mime};base64,{b64}'),
            ContentItem(text=f'[User sent an image. Answer directly from the image — do NOT call any tools.] {text}'),
        ]
    return query


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


def _run_turn(messages: list, llm, tool_definitions: list, tools: dict) -> str:
    """
    Run one LLM turn (with tool loop). Returns the final assistant text reply.
    Modifies messages in-place by appending assistant + tool result messages.
    """
    while True:
        # Collect full streamed response
        full_content = ''
        for chunks in llm.chat(messages=messages, functions=tool_definitions, stream=True):
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
            tool = tools.get(name)
            result = tool.call(json.dumps(args)) if tool else json.dumps({'error': f'Unknown tool: {name}'})
            messages.append({'role': 'function', 'name': name, 'content': result})


# --- Chat loop ---
def main():
    parser = argparse.ArgumentParser(description='Robot Navigation Assistant')
    parser.add_argument('--robot-ip',   default=os.environ.get('ROBOT_IP',      '127.0.0.1'))
    parser.add_argument('--robot-port', default=os.environ.get('ROBOT_PORT',    '5555'))
    parser.add_argument('--detection-mode', default=os.environ.get('DETECTION_MODE', 'sim'),
                        choices=['sim', 'real'])
    args = parser.parse_args()

    # Set env vars before importing tools (they read env at module load time)
    os.environ['ROBOT_IP']        = args.robot_ip
    os.environ['ROBOT_PORT']      = str(args.robot_port)
    os.environ['OBJECT_SERVER_IP'] = args.robot_ip
    os.environ['DETECTION_MODE']  = args.detection_mode

    from qwen_agent.llm import get_chat_model
    from tools.navigate_tool import NavigateToLandmark, ListLandmarks
    from tools.object_tool import GetDetectedObjects, ForgetObject
    from tools.motion_tool import SpinRobot
    from tools.detect_tool import ScanObjects, RegisterObjects, NavigateAndScan
    from tools.nutrition_tool import GetMealRecommendation

    llm = get_chat_model({
        'model': 'unsloth/Qwen3.5-9B-GGUF',
        'model_type': 'qwenvl_oai',
        'model_server': 'http://localhost:8080/v1',
        'api_key': 'EMPTY',
        'generate_cfg': {'thought_in_content': True},
    })

    tools = {
        'list_landmarks':               ListLandmarks(),
        'navigate_to_landmark':         NavigateToLandmark(),
        'get_detected_objects':         GetDetectedObjects(),
        'spin_robot':                   SpinRobot(),
        'forget_object':                ForgetObject(),
        'scan_objects':                 ScanObjects(),
        'register_objects':             RegisterObjects(),
        'navigate_and_scan':            NavigateAndScan(),
        'get_meal_recommendation':      GetMealRecommendation(),
    }

    print('Robot Navigation Assistant (type "exit" to quit)')
    print(f'  Robot IP      : {args.robot_ip}:{args.robot_port}')
    print(f'  Detection mode: {args.detection_mode}')
    print()

    messages = [SYSTEM_MSG]

    while True:
        query = input('User: ').strip()
        if query.lower() in ('exit', 'quit', ''):
            break

        messages.append({'role': 'user', 'content': _build_user_content(query)})

        print('Assistant: ', end='', flush=True)
        reply = _run_turn(messages, llm, TOOL_DEFINITIONS, tools)
        print(reply)
        print()


if __name__ == '__main__':
    main()
