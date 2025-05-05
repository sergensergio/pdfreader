from typing import Optional, Dict, Any
from groq import Groq


class LLM:
    def __init__(self, key: str) -> None:
        self.client = Groq(
            api_key=key,
        )

    def invoke(self, system_prompt: str, user_prompt: str, response_format: Optional[Dict[str, Any]] = None) -> str:
        completion = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            response_format=response_format,
            temperature=0
        )
        return completion.choices[0].message.content
