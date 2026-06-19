"""
Qwen Agent tool for nutritional meal recommendations.

Wraps nutri_rag's NutriAssistant to provide meal recommendations
based on what the user has eaten.

Requires:
    - nutri_rag LLM server running on localhost:8080
    - DuckDB knowledge base built (nutri_graph/data/nutri_kb.duckdb)
    - GAT embeddings trained (nutri_graph/outputs/embeddings/)
    - Text embeddings built (nutri_rag/data/embeddings/)
"""
import json
import os
import sys

import json5
from qwen_agent.tools.base import BaseTool, register_tool

# Add nutri_rag to import path
_MIMIR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_NUTRI_RAG_ROOT = os.path.join(_MIMIR_ROOT, 'nutri_rag')
if _NUTRI_RAG_ROOT not in sys.path:
    sys.path.insert(0, _NUTRI_RAG_ROOT)

from nutri_rag.assistant.pipeline import NutriAssistant  # noqa: E402

# Lazy singleton — initialized on first call to avoid slow startup
_assistant: NutriAssistant | None = None


def _get_assistant() -> NutriAssistant:
    global _assistant
    if _assistant is None:
        print('[nutrition] Initializing NutriAssistant (loading embeddings + DB)...')
        _assistant = NutriAssistant()
        print('[nutrition] NutriAssistant ready.')
    return _assistant


@register_tool('get_meal_recommendation')
class GetMealRecommendation(BaseTool):
    description = (
        'Get a personalized meal recommendation based on what the user has eaten. '
        'Uses nutritional gap analysis to suggest what to eat next. '
        'Example: user ate "an apple and a cup of milk" for breakfast, '
        'this tool recommends what to have for lunch.'
    )
    parameters = [
        {
            'name': 'eaten_foods',
            'type': 'string',
            'description': (
                'Description of what the user has eaten, '
                'e.g. "an apple and a cup of milk", "rice with chicken and broccoli".'
            ),
            'required': True,
        },
        {
            'name': 'eaten_meal_type',
            'type': 'string',
            'description': 'Which meal the eaten food was for: "breakfast", "lunch", "dinner", or "snack". Default: "breakfast".',
            'required': False,
        },
        {
            'name': 'next_meal',
            'type': 'string',
            'description': 'Which meal to recommend for: "breakfast", "lunch", "dinner", or "snack". Default: "lunch".',
            'required': False,
        },
        {
            'name': 'disliked_ingredients',
            'type': 'array',
            'description': (
                'Ingredients the user said they dislike or are allergic to, '
                'e.g. ["peanuts", "shellfish", "cilantro"]. Extract from the user '
                'message — only include what they actually stated. Omit if none mentioned.'
            ),
            'required': False,
        },
        {
            'name': 'health_condition',
            'type': 'string',
            'description': (
                'Chronic condition stated by the user that should shape macro targets: '
                '"diabetes" | "hypertension" | "obesity". Omit if not stated.'
            ),
            'required': False,
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json5.loads(params)
        eaten_foods = args.get('eaten_foods', '')
        eaten_meal_type = args.get('eaten_meal_type', 'breakfast')
        next_meal = args.get('next_meal', 'lunch')
        disliked = args.get('disliked_ingredients') or []
        health_condition = args.get('health_condition') or None

        if not eaten_foods.strip():
            return json.dumps({'status': 'error', 'message': 'Please describe what you have eaten.'})

        print(
            f'[nutrition] eaten="{eaten_foods}", meal_type={eaten_meal_type}, '
            f'next={next_meal}, disliked={disliked}, condition={health_condition}'
        )

        # Pull availability filter from env-configured source (none|json|zmq).
        # Default "none" preserves the current behavior with no filter.
        from nutri_rag.assistant.availability import get_available_fdc_ids
        available_fdc_ids = get_available_fdc_ids()
        if available_fdc_ids is not None:
            print(f'[nutrition] availability filter: {len(available_fdc_ids)} fdc_ids')

        try:
            assistant = _get_assistant()
            recommendation = assistant.recommend(
                user_message=f'What should I eat for {next_meal}?',
                eaten_description=eaten_foods,
                eaten_meal_type=eaten_meal_type,
                next_meal=next_meal,
                available_fdc_ids=available_fdc_ids,
                health_condition=health_condition,
                constraints={'disliked_ingredients': disliked} if disliked else None,
            )
            return json.dumps({
                'status': 'ok',
                'recommendation': recommendation,
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({'status': 'error', 'message': str(e)})
