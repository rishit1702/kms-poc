from abc import ABC, abstractmethod
from app.config import settings
import httpx


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt, user_prompt, context=""):
        pass


class OllamaProvider(LLMProvider):
    def generate(self, system_prompt, user_prompt, context=""):
        full_prompt = f"{system_prompt}\n\nKNOWLEDGE BASE CONTEXT:\n{context}\n\nUSER QUESTION:\n{user_prompt}"
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

    def generate(self, system_prompt, user_prompt, context=""):
        full_user = f"KNOWLEDGE BASE CONTEXT:\n{context}\n\nUSER QUESTION:\n{user_prompt}"
        r = self.client.chat.completions.create(
            model=settings.OPENROUTER_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": full_user}],
            max_tokens=1024,
        )
        return r.choices[0].message.content


class GeminiProvider(LLMProvider):
    def generate(self, system_prompt, user_prompt, context=""):
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        full_prompt = f"{system_prompt}\n\nKNOWLEDGE BASE CONTEXT:\n{context}\n\nUSER QUESTION:\n{user_prompt}"
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        return model.generate_content(full_prompt).text


class AnthropicProvider(LLMProvider):
    def __init__(self):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    def generate(self, system_prompt, user_prompt, context=""):
        full_user = f"KNOWLEDGE BASE CONTEXT:\n{context}\n\nUSER QUESTION:\n{user_prompt}"
        m = self.client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": full_user}],
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
