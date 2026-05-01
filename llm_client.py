# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# UNIFIED LLM CLIENT
# - Single API for text and vision (based on native ollama.chat).
# - Structured JSON using format=schema_pydantic + retry on validation error.
# - Basic JSON output sanitization (LLMs often leak markdown).
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from __future__ import annotations

import json
from typing import List, Optional, Type, TypeVar

import ollama
from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


class LLMClient:

    def __init__(self, base_url: str, default_model: str, json_max_retries: int = 2):
        self.base_url = base_url
        self.default_model = default_model
        self.json_max_retries = json_max_retries
        self._client = ollama.Client(host=base_url)

    # -- helpers ---------------------------------------------------------------------------------

    @staticmethod
    def _safe_json_extract(text: str) -> dict:
        """Tolerates markdown fences and model-leaked prefixes."""
        text = text.strip()
        # Strip code fences
        if text.startswith("```"):
            text = text.strip("`")
            # Remove potential "json" label
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            return json.loads(text[l : r + 1])
        raise ValueError("Could not parse JSON from model output.")

    # -- basic calls -----------------------------------------------------------------------------

    def chat_text(
        self,
        prompt: str,
        temperature: float,
        model: Optional[str] = None,
        images_b64: Optional[List[str]] = None,
    ) -> str:
        """Simple call; returns the text content of the response."""
        message = {"role": "user", "content": prompt}
        if images_b64:
            message["images"] = images_b64
        resp = self._client.chat(
            model=model or self.default_model,
            messages=[message],
            options={"temperature": temperature},
        )
        return resp["message"]["content"].strip()

    # -- structured JSON with Pydantic validation and retry --------------------------------------

    def chat_structured(
        self,
        prompt: str,
        schema: Type[T],
        temperature: float,
        model: Optional[str] = None,
        images_b64: Optional[List[str]] = None,
    ) -> T:
        """
        Calls the model requesting JSON, validates with Pydantic, and retries 
        in case of failure — including the error in the next message.
        """
        last_error: Optional[str] = None
        last_raw: Optional[str] = None

        for attempt in range(self.json_max_retries + 1):
            full_prompt = prompt
            if last_error and last_raw:
                full_prompt = (
                    prompt
                    + "\n\n---\nPrevious attempt failed validation:\n"
                    + f"INVALID_OUTPUT:\n{last_raw[:1500]}\n"
                    + f"ERROR:\n{last_error[:800]}\n"
                    + "Respond ONLY with valid JSON according to the schema, without markdown or comments."
                )

            message = {"role": "user", "content": full_prompt}
            if images_b64:
                message["images"] = images_b64

            try:
                resp = self._client.chat(
                    model=model or self.default_model,
                    messages=[message],
                    # format="json" forces JSON output on compatible Ollama models.
                    format="json",
                    options={"temperature": temperature},
                )
                raw = resp["message"]["content"]
                last_raw = raw
                data = self._safe_json_extract(raw)
                return schema.model_validate(data)
            except (ValueError, ValidationError, json.JSONDecodeError) as e:
                last_error = str(e)
                if attempt >= self.json_max_retries:
                    raise StructuredOutputError(
                        f"Failed to obtain valid JSON for {schema.__name__} "
                        f"after {self.json_max_retries + 1} attempts. "
                        f"Last error: {last_error}"
                    ) from e


class StructuredOutputError(RuntimeError):
    pass