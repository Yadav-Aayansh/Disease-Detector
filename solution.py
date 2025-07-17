# solution.py
import os
import requests

API_URL = "https://iit-ropar.truefoundry.cloud/api/llm/chat/completions"
MODEL = "openai-models/gpt-4o"

def get_disease_solution_translated(crop: str, disease: str, language: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an agricultural assistant for Indian farmers. "
                    "Given a crop and its disease, explain in the requested language using simple words suitable for rural farmers: "
                    "1) What is the disease, 2) Why it happens (cause), 3) How to recognize it (symptoms), "
                    "4) What to do (solution: chemical and organic if possible), 5) How to prevent it, and "
                    "6) Any tips for early detection or monitoring. "
                    "Respond only in the requested language. No markdown."
                )
            },
            {
                "role": "user",
                "content": (
                    f"My {crop} crop is affected by {disease}. "
                    f"Give full advice in {language}."
                )
            }
        ],
        "tools": [
        {
            "type": "function",
            "function": {"name": "web_search"}
        }
    ]
    }

    HEADERS = {
    "Authorization": f"Bearer {os.getenv('TOKEN')}",
    "Content-Type": "application/json"
    }

    response = requests.post(API_URL, json=payload, headers=HEADERS)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
