"""LiteLLM integration provider implementing structured response outputs."""

import logging

import litellm
from pydantic import BaseModel, Field

from autobacktest.config import settings
from autobacktest.llm.base import AgentContext, AgentEdit, LLMError, LLMProvider
from autobacktest.llm.prompts import build_messages


class AgentEditResponse(BaseModel):
    """Pydantic schema used for structured output parsing via LiteLLM."""

    strategy_code: str = Field(description="The complete new Python strategy source code.")
    config_yaml: str = Field(description="The complete new YAML parameters configuration.")
    reasoning: str = Field(description="Quantitative reasoning and explanation for changes.")
    lessons_text: str | None = Field(
        default=None,
        description="Complete updated lessons learned markdown text when changed.",
    )


logger = logging.getLogger(__name__)


def _compute_run_max_tokens(
    model: str,
    messages: list[dict[str, object]],
    env_limit: int,
    instance_max_tokens: int | None,
) -> int:
    """Compute the effective ``max_tokens`` for a single LLM completion call.

    Respects the configured limit while capping by remaining context window
    headroom.  An explicit *instance_max_tokens* overrides *env_limit* when
    the two differ (used for per-call retry overrides).

    Args:
        model: LiteLLM model identifier.
        messages: Prompt messages list (used for token counting).
        env_limit: Global ``llm_max_tokens`` setting from the environment.
        instance_max_tokens: Per-instance override, or ``None`` to use *env_limit*.

    Returns:
        int: Positive effective token budget.
    """
    try:
        prompt_tokens = litellm.token_counter(model=model, messages=messages)
    except Exception:
        prompt_tokens = len(str(messages)) // 4

    try:
        context_window = litellm.get_max_tokens(model) or 128_000
    except Exception:
        context_window = 128_000

    buffer = 4096
    dynamic_max = context_window - prompt_tokens - buffer
    cap = instance_max_tokens if instance_max_tokens is not None and instance_max_tokens != env_limit else env_limit
    return max(1, min(dynamic_max, cap))  # type: ignore[no-any-return]


def _extract_clean_json(content: str) -> str:
    r"""Strip Markdown code fences and extract the first JSON object from *content*.

    Handles ``\`\`\`json``, bare ``\`\`\```, and brace-delimited fallback extraction.

    Args:
        content: Raw LLM response content string.

    Returns:
        str: Content trimmed to a JSON object, or the original stripped string.
    """
    clean = content.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
        if clean.endswith("```"):
            clean = clean[:-3]
    elif clean.startswith("```"):
        clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
    clean = clean.strip()
    if not (clean.startswith("{") and clean.endswith("}")):
        start_idx = clean.find("{")
        end_idx = clean.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            clean = clean[start_idx : end_idx + 1]
    return clean


def _extract_usage_stats(
    response: object,
) -> tuple[int, int, int, int, float]:
    """Extract token counts and cost from a LiteLLM completion response.

    Args:
        response: LiteLLM ``ModelResponse`` object.

    Returns:
        tuple: ``(prompt_tokens, completion_tokens, total_tokens, cached_tokens, cost)``.
    """
    usage = getattr(response, "usage", None)
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    total_tokens = usage.total_tokens if usage else 0

    cached_tokens = 0
    if usage:
        try:
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
            else:
                cached_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        except Exception:
            pass

    try:
        cost = litellm.completion_cost(completion_response=response) or 0.0
    except Exception:
        cost = 0.0

    return prompt_tokens, completion_tokens, total_tokens, cached_tokens, cost


def _classify_llm_error(e: Exception) -> tuple[str | None, bool]:
    """Return ``(finish_reason, retryable)`` for an exception caught during generation.

    Args:
        e: The exception to classify.

    Returns:
        tuple: ``finish_reason`` string or ``None``, and ``retryable`` bool.
    """
    finish_reason: str | None = None
    retryable = True

    if isinstance(e, ValueError) and "stopped prematurely" in str(e):
        if "finish_reason: length" in str(e):
            finish_reason = "length"
    elif "length" in str(e).lower() or getattr(e, "finish_reason", None) == "length":
        finish_reason = "length"

    if isinstance(
        e,
        (
            litellm.BadRequestError,  # type: ignore[attr-defined]
            litellm.AuthenticationError,  # type: ignore[attr-defined]
            litellm.NotFoundError,  # type: ignore[attr-defined]
        ),
    ) or getattr(e, "status_code", None) in (400, 401, 403, 404):
        retryable = False

    return finish_reason, retryable


class LiteLLMProvider(LLMProvider):
    """Concrete LLM Provider using LiteLLM for structured code edits."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        """Initialize the provider with model and completion parameters.

        Args:
            model: The LiteLLM model identifier (e.g. "openai/gpt-4o").
            temperature: Sampling temperature between 0.0 and 1.0.
            max_tokens: Token limit for LLM generation.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        if settings.litellm_debug:
            litellm._turn_on_debug()  # type: ignore[attr-defined, no-untyped-call]

        try:
            self.supports_schema = litellm.supports_response_schema(model=self.model)
        except Exception:
            self.supports_schema = False

        try:
            self.supports_cache = litellm.supports_prompt_caching(model=self.model)
        except Exception:
            self.supports_cache = False

    @property
    def provider_name(self) -> str:
        """Return the unique string identification of the provider."""
        return "litellm"

    def generate_edit(self, context: AgentContext) -> AgentEdit:
        """Call LiteLLM API to generate a structured strategy modification.

        Args:
            context: Immutable optimization loop context.

        Returns:
            AgentEdit: The parsed code and config updates.

        Raises:
            LLMError: If litellm API, structured output parsing, or request fails.
        """
        cache_enabled = self.supports_cache and settings.llm_prompt_cache
        messages = build_messages(context, cache_supported=cache_enabled)

        env_limit = getattr(settings, "llm_max_tokens", 4096)
        instance_override = self.max_tokens if self.max_tokens != env_limit else None
        run_max_tokens = _compute_run_max_tokens(self.model, messages, env_limit, instance_override)

        try:
            resp_format = AgentEditResponse if self.supports_schema else {"type": "json_object"}
            response = litellm.completion(
                model=self.model,
                messages=messages,
                response_format=resp_format,
                temperature=self.temperature,
                max_tokens=run_max_tokens,
                request_timeout=settings.llm_request_timeout,
            )

            if not response.choices:
                raise ValueError("LLM returned no choices.")

            choice = response.choices[0]
            content = choice.message.content
            finish_reason = getattr(choice, "finish_reason", None)

            if finish_reason == "length" or not content:
                length_or_zero = len(content) if content else 0
                raise ValueError(
                    f"LLM generation stopped prematurely. "
                    f"finish_reason: {finish_reason}. "
                    f"content_length: {length_or_zero}"
                )

            parsed_response = AgentEditResponse.model_validate_json(_extract_clean_json(content))
            prompt_tokens, completion_tokens, total_tokens, cached_tokens, cost = _extract_usage_stats(response)

            logger.debug(
                "LLM call complete: prompt=%d completion=%d cached=%d cost=$%.4f",
                prompt_tokens,
                completion_tokens,
                cached_tokens,
                cost,
            )

            return AgentEdit(
                strategy_code=parsed_response.strategy_code,
                config_yaml=parsed_response.config_yaml,
                reasoning=parsed_response.reasoning,
                raw_response=content,
                lessons_text=parsed_response.lessons_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=cost,
                cached_tokens=cached_tokens,
            )

        except Exception as e:
            finish_reason, retryable = _classify_llm_error(e)
            raise LLMError(
                provider="litellm",
                model=self.model,
                detail=str(e),
                retryable=retryable,
                finish_reason=finish_reason,
            ) from e
