from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from app.config import settings
import httpx


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt, user_prompt, context="", history: Optional[List[Dict]] = None):
        """
        Generate a response.

        history: Optional list of past turns, each like {"role": "user"|"assistant", "content": "..."}.
                 The current user_prompt is the LATEST user turn and is NOT included in history.
                 Pass None or [] for single-shot (no memory) calls — used by query expansion etc.
        """
        pass


def _format_history_as_text(history: Optional[List[Dict]]) -> str:
    """For providers that take a single prompt string (Ollama, Gemini), flatten history into text."""
    if not history:
        return ""
    lines = []
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


class OllamaProvider(LLMProvider):
    def generate(self, system_prompt, user_prompt, context="", history: Optional[List[Dict]] = None):
        history_text = _format_history_as_text(history)
        history_block = f"\n\nCONVERSATION HISTORY:\n{history_text}" if history_text else ""
        full_prompt = (
            f"{system_prompt}\n\n"
            f"KNOWLEDGE BASE CONTEXT:\n{context}"
            f"{history_block}\n\n"
            f"CURRENT USER QUESTION:\n{user_prompt}"
        )
        with httpx.Client(timeout=300.0) as client:
            r = client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={"model": settings.OLLAMA_MODEL, "prompt": full_prompt, "stream": False},
            )
            r.raise_for_status()
            return r.json()["response"]


class OpenRouterProvider(LLMProvider):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=settings.OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

    def generate(self, system_prompt, user_prompt, context="", history: Optional[List[Dict]] = None):
        # OpenAI-compatible chat API natively supports history as a list of messages.
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for turn in history:
                role = turn.get("role", "user")
                if role not in ("user", "assistant"):
                    continue
                messages.append({"role": role, "content": turn.get("content", "")})
        # The current turn carries the retrieved context plus the new user question.
        messages.append({
            "role": "user",
            "content": f"KNOWLEDGE BASE CONTEXT:\n{context}\n\nCURRENT USER QUESTION:\n{user_prompt}",
        })
        r = self.client.chat.completions.create(
            model=settings.OPENROUTER_MODEL,
            messages=messages,
            max_tokens=1024,
        )
        return r.choices[0].message.content


class GeminiProvider(LLMProvider):
    def generate(self, system_prompt, user_prompt, context="", history: Optional[List[Dict]] = None):
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        history_text = _format_history_as_text(history)
        history_block = f"\n\nCONVERSATION HISTORY:\n{history_text}" if history_text else ""
        full_prompt = (
            f"{system_prompt}\n\n"
            f"KNOWLEDGE BASE CONTEXT:\n{context}"
            f"{history_block}\n\n"
            f"CURRENT USER QUESTION:\n{user_prompt}"
        )
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        return model.generate_content(full_prompt).text


class AnthropicProvider(LLMProvider):
    def __init__(self):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    def generate(self, system_prompt, user_prompt, context="", history: Optional[List[Dict]] = None):
        # Anthropic also supports history as a messages list.
        messages = []
        if history:
            for turn in history:
                role = turn.get("role", "user")
                if role not in ("user", "assistant"):
                    continue
                messages.append({"role": role, "content": turn.get("content", "")})
        messages.append({
            "role": "user",
            "content": f"KNOWLEDGE BASE CONTEXT:\n{context}\n\nCURRENT USER QUESTION:\n{user_prompt}",
        })
        m = self.client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )
        return m.content[0].text


def get_llm():
    p = settings.LLM_PROVIDER
    if p == "ollama":
        return OllamaProvider()
    if p == "openrouter":
        return OpenRouterProvider()
    if p == "gemini":
        return GeminiProvider()
    if p == "anthropic":
        return AnthropicProvider()
    raise ValueError(f"Unknown LLM_PROVIDER: {p}")
