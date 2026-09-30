from os import getenv
import anthropic

MODEL = getenv("ANTHROPIC_MODEL", "claude-opus-5-5")


class LLMUnavailable(Exception):
    """No Anthropic credentials configured, or the model refused the request."""


_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic()
    return _client


async def generate_answer(system: str, user_message: str) -> str:
    if not getenv("ANTHROPIC_API_KEY"):
        raise LLMUnavailable("ANTHROPIC_API_KEY is not set in backend/.env.")

    response = await _get_client().beta.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user_message}],
        output_config={"effort": "low"},
        # If the primary model declines, the API re-runs the request on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )

    if response.stop_reason == "refusal":
        raise LLMUnavailable("The model declined to answer this request.")
    return "".join(b.text for b in response.content if b.type == "text").strip()
