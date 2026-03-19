from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")  # 别再用 mini 这种弱智模型了
DEFAULT_DEEPSEEK_CHAT_MODEL = os.environ.get("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
DEFAULT_DEEPSEEK_REASONER_MODEL = os.environ.get("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner")


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
                raise ValueError(
                    "没有配置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY！别想再用 Mock 假数据糊弄，立刻去配环境变量！")

        if self.provider == "deepseek":
            self.model = self.model or self.deepseek_chat_model
        elif self.provider == "openai":
            self.model = self.model or DEFAULT_MODEL

    def _set_last_result(self, result: LLMResult) -> LLMResult:
        self.last_result = result
        usage = result.usage or {}
        if result.provider in {"deepseek", "openai"} and usage:
            prompt_toks = usage.get("prompt_tokens")
            completion_toks = usage.get("completion_tokens")
            total_toks = usage.get("total_tokens")
            reasoning_toks = ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
            extras = []
            if prompt_toks is not None:
                extras.append(f"prompt={prompt_toks}")
            if completion_toks is not None:
                extras.append(f"completion={completion_toks}")
            if total_toks is not None:
                extras.append(f"total={total_toks}")
            if reasoning_toks is not None:
                extras.append(f"reasoning={reasoning_toks}")
            if extras:
                print(f"[LLM_USAGE] provider={result.provider} model={result.model} " + " ".join(extras))
        return result

    @staticmethod
    def _extract_openai_output_text(payload: Dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError(f"LLM response missing choices: {payload}")
        return choices[0].get("message", {}).get("content", "").strip()

    @staticmethod
    def _extract_chat_text(payload: Dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError(f"LLM response missing choices: {payload}")
        content = choices[0].get("message", {}).get("content", "")
        return content.strip() if isinstance(content, str) else str(content)

    @staticmethod
    def _extract_deepseek_reasoning(payload: Dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        reasoning = choices[0].get("message", {}).get("reasoning_content", "")
        return reasoning.strip() if isinstance(reasoning, str) else ""

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

    def _call_openai_responses(self, prompt: str, system_prompt: str, expect_json: bool = False, max_tokens: int = 4000,
                               temperature: float = 0.6) -> LLMResult:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set for provider=openai")
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if expect_json:
            body["response_format"] = {"type": "json_object"}

        payload = self._post_json(f"{self.openai_base_url}/chat/completions", headers, body)
        text = self._extract_openai_output_text(payload)
        usage = payload.get("usage") or {}
        return self._set_last_result(
            LLMResult(text=text, provider="openai", model=self.model or DEFAULT_MODEL, raw=payload, usage=usage))

    def _call_deepseek_chat_completion(
            self,
            prompt: str,
            system_prompt: str,
            expect_json: bool = False,
            use_reasoner: bool = False,
            max_tokens: int = 4000,
            temperature: float = 0.6,
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
        return self._set_last_result(
            LLMResult(text=text, provider="deepseek", model=model, raw=payload, usage=usage, reasoning=reasoning))

    def generate_result(
            self,
            prompt: str,
            system_prompt: str,
            expect_json: bool = False,
            max_tokens: int = 4000,
            temperature: float = 0.6,
            use_reasoner: bool = False,
    ) -> LLMResult:
        if self.provider == "openai":
            return self._call_openai_responses(prompt, system_prompt, expect_json, max_tokens, temperature)
        if self.provider == "deepseek":
            return self._call_deepseek_chat_completion(
                prompt,
                system_prompt,
                expect_json=expect_json,
                use_reasoner=use_reasoner,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(
            self,
            prompt: str,
            system_prompt: str,
            max_tokens: int = 4000,
            temperature: float = 0.6,
            use_reasoner: bool = False,
    ) -> str:
        return self.generate_result(
            prompt,
            system_prompt,
            expect_json=False,
            max_tokens=max_tokens,
            temperature=temperature,
            use_reasoner=use_reasoner,
        ).text

    def generate_json(
            self,
            prompt: str,
            system_prompt: str,
            max_tokens: int = 2000,
            temperature: float = 0.1,  # JSON 提取时温度可以低一点，防止格式错乱
    ) -> Dict[str, Any]:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                result = self.generate_result(
                    prompt,
                    system_prompt,
                    expect_json=True,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    use_reasoner=False,  # JSON输出通常不需要reasoner
                )
                text = (result.text or "").strip()
                if not text:
                    raise ValueError("LLM returned empty content in JSON mode")
                return self._extract_json(text)
            except Exception as exc:
                last_exc = exc
                print(f"[LLM_JSON_RETRY] attempt={attempt} reason=bad_json error={exc}")
                continue
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        # 使用字符乘法和正则量词彻底阻断 Markdown 渲染器解析连续三个反引号
        magic_fence = "`" * 3
        if cleaned.startswith(magic_fence):
            cleaned = re.sub(r"^`{3}(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*`{3}$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
            raise TypeError(f"Expected a JSON object, got {type(parsed)}")
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise ValueError(f"无法从中解析出JSON: {cleaned[:100]}...")
            parsed = json.loads(match.group(0))
            if not isinstance(parsed, dict):
                raise TypeError(f"Expected a JSON object, got {type(parsed)}")
            return parsed
