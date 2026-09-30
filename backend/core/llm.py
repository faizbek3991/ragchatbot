from os import getenv
from google import genai
from google.genai import errors, types

MODEL = getenv("GEMINI_MODEL", "gemini-2.5-flash")


class LLMUnavailable(Exception):
    """No Gemini credentials configured, or the model call failed or was blocked."""


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=getenv("GEMINI_API_KEY"))
    return _client


async def generate_answer(system: str, user_message: str, history: list[dict] | None = None) -> str:
    if not getenv("GEMINI_API_KEY"):
        raise LLMUnavailable("GEMINI_API_KEY is not set in backend/.env.")

    # Gemini calls the assistant role "model".
    contents = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part(text=m["content"])],
        )
        for m in [*(history or []), {"role": "user", "content": user_message}]
    ]

    try:
        response = await _get_client().aio.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.2,  # low: stay close to the passages
                max_output_tokens=2048,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
    except errors.APIError as e:
        raise LLMUnavailable(f"Gemini API error ({e.code}): {e.message}")

    text = (response.text or "").strip()
    if not text:
        feedback = response.prompt_feedback
        reason = feedback.block_reason if feedback and feedback.block_reason else "empty response"
        raise LLMUnavailable(f"Gemini returned no answer ({reason}).")
    return text
