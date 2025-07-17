# create_events.py

import json
from typing import List
from ai_helper import generate_prediction_event

def load_categories() -> dict:
    with open("categories.json", "r", encoding="utf-8") as f:
        return json.load(f)

def generate_suggested_events(country: str, market: str, subcategory: str) -> List[str]:
    return generate_prediction_event(country, market, subcategory)