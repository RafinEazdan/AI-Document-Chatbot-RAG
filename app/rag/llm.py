"""Gemini LLM provider."""

from typing import List, Dict

import google.generativeai as genai

from app.core.config import Config
from app.core.interfaces import ILLMProvider


class GeminiProvider(ILLMProvider):
    """ILLMProvider backed by the Google Gemini API."""

    def __init__(self, config: Config) -> None:
        genai.configure(api_key=config.GEMINI_API_KEY)
        self._model_name = config.GEMINI_MODEL

    def complete(self, messages: List[Dict[str, str]]) -> str:
        system_parts = []
        history = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                history.append({"role": "model", "parts": [content]})

        system_instruction = "\n".join(system_parts) if system_parts else None
        model = genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=system_instruction,
        )

        if len(history) > 1:
            chat = model.start_chat(history=history[:-1])
            response = chat.send_message(history[-1]["parts"][0])
        else:
            response = model.generate_content(
                history[0]["parts"][0] if history else ""
            )

        return response.text.strip()
