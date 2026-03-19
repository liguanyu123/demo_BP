from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

from problem_description import build_problem_description
from prompt_library import (
    MOCK_ERROR_ANALYSIS,
    MOCK_PROBLEM_DESCRIPTION_JUDGE_OK,
    MOCK_SOLVER_JUDGE_OK,
)


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
DEFAULT_DEEPSEEK_CHAT_MODEL = os.environ.get("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
DEFAULT_DEEPSEEK_REASONER_MODEL = os.environ.get("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner")


MOCK_POLICY_CODE = """
from trusted_solver import solve_with_policy


def order_items(items):
    def rank(item_id):
        item = items[item_id]
        return (
            1 if item.get("fragile", False) else 0,
            -(item["l"] * item["w"]),
            -(item["l"] * item["w"] * item["h"]),
            -item["h"],
            item_id,
        )
    return sorted(items.keys(), key=rank)


def score_candidate(item_id, item, candidate, state, bin_size):
    return (
        candidate["z"],
        -candidate["support_area"],
        candidate["used_height_after"],
        candidate["bbox_volume_after"],
        1 if item.get("fragile", False) else 0,
        -(item["l"] * item["w"]),
        -item["h"],
        candidate["y"],
        candidate["x"],
        item_id,
    )


def solve_packing(items, bin_size):
    return solve_with_policy(items, bin_size, order_items, score_candidate, heavy=(len(items) >= 20))
""".strip()


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    raw: Optional[Dict[str, Any]] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


class LLMClient:
    def __init__(
        self,
        provider: str = "auto",
        model: Optional[str] = None,
        timeout: int = 180,
        retries: int = 3,
    ):
        self.provider = provider
        self.model = model
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.last_result: Optional[LLMResult] = None

        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.openai_base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL).rstrip("/")

        self.deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.deepseek_base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL).rstrip("/")
        self.deepseek_chat_model = DEFAULT_DEEPSEEK_CHAT_MODEL
        self.deepseek_reasoner_model = DEFAULT_DEEPSEEK_REASONER_MODEL

        if self.provider == "auto":
            if self.deepseek_api_key:
                self.provider = "deepseek"
            elif self.openai_api_key:
                self.provider = "openai"
            else:
                self.provider = "mock_reference"

        if self.provider == "deepseek":
            self.model = self.model or self.deepseek_chat_model
        elif self.provider == "openai":
            self.model = self.model or DEFAULT_MODEL
        else:
            self.model = self.model or "trusted-template"

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock_reference"

    def _set_last_result(self, result: LLMResult) -> LLMResult:
        self.last_result = result
        usage = result.usage or {}
        if result.provider in {"deepseek", "openai"} and usage:
            prompt_toks = usage.get("prompt_tokens")
            completion_toks = usage.get("completion_tokens")
            total_toks = usage.get("total_tokens")
            reasoning_toks = ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
            cache_hit_toks = usage.get("prompt_cache_hit_tokens")
            cache_miss_toks = usage.get("prompt_cache_miss_tokens")
            extras = []
            if prompt_toks is not None:
                extras.append(f"prompt={prompt_toks}")
            if completion_toks is not None:
                extras.append(f"completion={completion_toks}")
            if total_toks is not None:
                extras.append(f"total={total_toks}")
            if reasoning_toks is not None:
                extras.append(f"reasoning={reasoning_toks}")
            if cache_hit_toks is not None:
                extras.append(f"cache_hit={cache_hit_toks}")
            if cache_miss_toks is not None:
                extras.append(f"cache_miss={cache_miss_toks}")
            if extras:
                print(f"[LLM_USAGE] provider={result.provider} model={result.model} " + " ".join(extras))
        return result

    def get_last_meta(self) -> Dict[str, Any]:
        if self.last_result is None:
            return {}
        reasoning_preview = self.last_result.reasoning[:1000] if self.last_result.reasoning else ""
        return {
            "provider": self.last_result.provider,
            "model": self.last_result.model,
            "usage": self.last_result.usage,
            "reasoning_preview": reasoning_preview,
        }

    def _mock_generate(self, role_hint: Optional[str], context: Optional[Dict[str, Any]]) -> LLMResult:
        if role_hint == "problem_description_generate":
            if not context or "items" not in context or "bin_size" not in context:
                raise ValueError("Mock problem description generation requires items and bin_size context")
            text = json.dumps(
                build_problem_description(context["items"], context["bin_size"]),
                ensure_ascii=False,
                indent=2,
            )
        elif role_hint == "problem_description_judge":
            text = json.dumps(MOCK_PROBLEM_DESCRIPTION_JUDGE_OK, ensure_ascii=False)
        elif role_hint == "problem_description_revise":
            if not context or "candidate_description" not in context:
                raise ValueError("Mock description revision requires candidate_description context")
            text = json.dumps(context["candidate_description"], ensure_ascii=False, indent=2)
        elif role_hint == "solver_generate":
            text = MOCK_POLICY_CODE
        elif role_hint == "solver_revise":
            text = MOCK_POLICY_CODE
        elif role_hint == "solver_judge":
            text = json.dumps(MOCK_SOLVER_JUDGE_OK, ensure_ascii=False)
        elif role_hint == "error_analysis":
            text = json.dumps(MOCK_ERROR_ANALYSIS, ensure_ascii=False)
        else:
            text = ""
        return self._set_last_result(LLMResult(text=text, provider="mock_reference", model="trusted-template", raw=None))

    @staticmethod
    def _extract_openai_output_text(payload: Dict[str, Any]) -> str:
        if isinstance(payload.get("output_text"), str):
            return payload["output_text"]
        parts = []
        for item in payload.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"} and "text" in content:
                    parts.append(content["text"])
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _extract_chat_text(payload: Dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError(f"LLM response missing choices: {payload}")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for entry in content:
                if isinstance(entry, dict) and isinstance(entry.get("text"), str):
                    parts.append(entry["text"])
                elif isinstance(entry, str):
                    parts.append(entry)
            return "\n".join(part for part in parts if part).strip()
        raise ValueError(f"Unsupported chat content format: {content!r}")

    @staticmethod
    def _extract_deepseek_reasoning(payload: Dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        reasoning = message.get("reasoning_content", "")
        if isinstance(reasoning, str):
            return reasoning.strip()
        return ""

    def _post_json(self, url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                if attempt == self.retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 4))
        assert last_exc is not None
        raise RuntimeError(f"LLM request failed after {self.retries} attempt(s): {last_exc}") from last_exc

    def _call_openai_responses(self, prompt: str, system_prompt: str) -> LLMResult:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set for provider=openai")
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "instructions": system_prompt,
            "input": prompt,
        }
        payload = self._post_json(f"{self.openai_base_url}/responses", headers, body)
        text = self._extract_openai_output_text(payload).strip()
        usage = payload.get("usage") or {}
        return self._set_last_result(LLMResult(text=text, provider="openai", model=self.model or DEFAULT_MODEL, raw=payload, usage=usage))

    def _call_deepseek_chat_completion(
        self,
        prompt: str,
        system_prompt: str,
        expect_json: bool = False,
        use_reasoner: bool = False,
        max_tokens: int = 4000,
        temperature: float = 0.2,
    ) -> LLMResult:
        if not self.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set for provider=deepseek")
        model = self.deepseek_reasoner_model if use_reasoner else (self.model or self.deepseek_chat_model)
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if expect_json:
            body["response_format"] = {"type": "json_object"}
        payload = self._post_json(f"{self.deepseek_base_url}/chat/completions", headers, body)
        text = self._extract_chat_text(payload)
        usage = payload.get("usage") or {}
        reasoning = self._extract_deepseek_reasoning(payload)
        return self._set_last_result(LLMResult(text=text, provider="deepseek", model=model, raw=payload, usage=usage, reasoning=reasoning))

    def generate_result(
        self,
        prompt: str,
        system_prompt: str,
        role_hint: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4000,
        temperature: float = 0.2,
        use_reasoner: bool = False,
    ) -> LLMResult:
        if self.is_mock:
            return self._mock_generate(role_hint, context)
        if self.provider == "openai":
            return self._call_openai_responses(prompt, system_prompt)
        if self.provider == "deepseek":
            return self._call_deepseek_chat_completion(
                prompt,
                system_prompt,
                expect_json=False,
                use_reasoner=use_reasoner,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        role_hint: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4000,
        temperature: float = 0.2,
        use_reasoner: bool = False,
    ) -> str:
        return self.generate_result(
            prompt,
            system_prompt,
            role_hint=role_hint,
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
            use_reasoner=use_reasoner,
        ).text

    def generate_json(
        self,
        prompt: str,
        system_prompt: str,
        role_hint: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        use_reasoner: bool = False,
    ) -> Dict[str, Any]:
        if self.provider == "deepseek" and not self.is_mock:
            # DeepSeek JSON mode can occasionally return empty content.
            # For structured sub-agents, prefer deepseek-chat over the reasoner and retry a few times.
            last_exc: Optional[Exception] = None
            for attempt in range(1, self.retries + 1):
                result = self._call_deepseek_chat_completion(
                    prompt,
                    system_prompt,
                    expect_json=True,
                    use_reasoner=False,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = (result.text or "").strip()
                if not text:
                    last_exc = ValueError("DeepSeek returned empty content in JSON mode")
                    print(f"[LLM_JSON_RETRY] provider=deepseek attempt={attempt} reason=empty_content")
                    continue
                try:
                    return self._extract_json(text)
                except Exception as exc:
                    last_exc = exc
                    preview = text[:200].replace("\n", " ")
                    print(f"[LLM_JSON_RETRY] provider=deepseek attempt={attempt} reason=bad_json preview={preview!r}")
                    continue
            assert last_exc is not None
            raise last_exc
        text = self.generate(
            prompt,
            system_prompt,
            role_hint=role_hint,
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
            use_reasoner=use_reasoner,
        )
        return self._extract_json(text)

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
            raise TypeError(f"Expected a JSON object, got {type(parsed)}")
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise
            parsed = json.loads(match.group(0))
            if not isinstance(parsed, dict):
                raise TypeError(f"Expected a JSON object, got {type(parsed)}")
            return parsed


if __name__ == "__main__":
    client = LLMClient(provider="auto")
    print(client.generate("Say hello in JSON.", "Return only JSON.", role_hint="solver_judge"))
