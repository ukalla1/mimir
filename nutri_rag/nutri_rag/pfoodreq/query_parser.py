"""Parse PFoodReq benchmark examples into structured constraints.

Each PFoodReq example contains pre-structured fields — no NLP is needed.
This module extracts them into a clean format for the retriever.
"""

from __future__ import annotations


def parse_example(example: dict) -> dict:
    """Parse a single PFoodReq JSONL example into structured constraints.

    Returns:
        {
            "qid": str,
            "query_text": str,           # expanded query text
            "original_query": str,       # original query without persona
            "tag": str,                  # cuisine/category tag name
            "tag_uri": str,              # FoodKG tag URI
            "positive_ingredients": [],  # ingredients user likes / must include
            "negative_ingredients": [],  # ingredients user dislikes / must exclude
            "nutrient_constraints": [],  # [{nutrient, level, range}]
            "guidelines": {},            # raw guideline dict
            "ground_truth": [],          # list of recipe names (answers)
            "origin_answers": [],        # recipes matching original query only
            "domain_type": str,          # "in-domain" or "out-of-domain"
        }
    """
    # Tag extraction
    entities = example.get("entities", [])
    tag_name = entities[0][0] if entities else ""
    tag_uri = ""
    topic_keys = example.get("topicKey", [])
    if topic_keys:
        tag_uri = topic_keys[0]

    # Persona: ingredient likes/dislikes
    # constrained_entities["2"] contains ALL negative ingredients
    # (persona dislikes + original query exclusions)
    persona = example.get("persona", {})
    constrained = persona.get("constrained_entities", {})
    positive_ingredients = persona.get("ingredient_likes", [])
    negative_ingredients = list(constrained.get("2", persona.get("ingredient_dislikes", [])))

    # Nutrient constraints from explicit_nutrition
    nutrient_constraints = []
    for nc in example.get("explicit_nutrition", []):
        nutrient_constraints.append({
            "nutrient": nc.get("nutrition", ""),
            "level": nc.get("level", ""),
            "range": nc.get("range", []),
        })

    return {
        "qid": example.get("qId", ""),
        "query_text": example.get("qText", ""),
        "original_query": example.get("qOriginText", ""),
        "tag": tag_name,
        "tag_uri": tag_uri,
        "positive_ingredients": positive_ingredients,
        "negative_ingredients": negative_ingredients,
        "nutrient_constraints": nutrient_constraints,
        "guidelines": example.get("guideline", {}),
        "ground_truth": example.get("answers", []),
        "origin_answers": example.get("origin_answers", []),
        "domain_type": example.get("domainType", ""),
    }


def load_examples(path: str) -> list[dict]:
    """Load PFoodReq JSONL file and parse all examples."""
    import json

    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            examples.append(parse_example(raw))
    return examples
