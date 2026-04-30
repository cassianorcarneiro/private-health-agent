# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Cliente LLM unificado.
# - Uma única API para texto e visão (baseada em ollama.chat nativo).
# - JSON estruturado com format=schema_pydantic + retry com erro de validação.
# - Sanitização básica de saída JSON (LLMs vazam markdown).
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
        """Tolera markdown fences e prefixos vazados do modelo."""
        text = text.strip()
        # Strip code fences
        if text.startswith("```"):
            text = text.strip("`")
            # Remove eventual rótulo "json"
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

    # -- chamadas básicas ------------------------------------------------------------------------

    def chat_text(
        self,
        prompt: str,
        temperature: float,
        model: Optional[str] = None,
        images_b64: Optional[List[str]] = None,
    ) -> str:
        """Chamada simples; retorna o conteúdo de texto da resposta."""
        message = {"role": "user", "content": prompt}
        if images_b64:
            message["images"] = images_b64
        resp = self._client.chat(
            model=model or self.default_model,
            messages=[message],
            options={"temperature": temperature},
        )
        return resp["message"]["content"].strip()

    # -- JSON estruturado com validação Pydantic e retry -----------------------------------------

    def chat_structured(
        self,
        prompt: str,
        schema: Type[T],
        temperature: float,
        model: Optional[str] = None,
        images_b64: Optional[List[str]] = None,
    ) -> T:
        """
        Chama o modelo pedindo JSON, valida com Pydantic, e tenta novamente
        em caso de falha — incluindo o erro na próxima mensagem.
        """
        last_error: Optional[str] = None
        last_raw: Optional[str] = None

        for attempt in range(self.json_max_retries + 1):
            full_prompt = prompt
            if last_error and last_raw:
                full_prompt = (
                    prompt
                    + "\n\n---\nTentativa anterior falhou validação:\n"
                    + f"OUTPUT_INVÁLIDO:\n{last_raw[:1500]}\n"
                    + f"ERRO:\n{last_error[:800]}\n"
                    + "Responda APENAS o JSON válido conforme o schema, sem markdown nem comentários."
                )

            message = {"role": "user", "content": full_prompt}
            if images_b64:
                message["images"] = images_b64

            try:
                resp = self._client.chat(
                    model=model or self.default_model,
                    messages=[message],
                    # format="json" força saída JSON em modelos compatíveis com Ollama.
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
                        f"Falha ao obter JSON válido para {schema.__name__} "
                        f"após {self.json_max_retries + 1} tentativas. "
                        f"Último erro: {last_error}"
                    ) from e


class StructuredOutputError(RuntimeError):
    pass
