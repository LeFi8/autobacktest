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
        messages = build_messages(context)

        # 1. Centralized Dynamic Token Allocation
        try:
            prompt_tokens = litellm.token_counter(model=self.model, messages=messages)
        except Exception:
            prompt_tokens = len(str(messages)) // 4

        try:
            context_window = litellm.get_max_tokens(self.model) or 128000
        except Exception:
            context_window = 128000

        buffer = 4096
        dynamic_max = context_window - prompt_tokens - buffer

        # Respect the configured max tokens limit, but cap it by the remaining context window
        env_limit = getattr(settings, "llm_max_tokens", 4096)
        run_max_tokens = max(1, min(dynamic_max, env_limit))

        # Respect customized instance max_tokens if explicitly modified (e.g. retry or constructor test override)
        if hasattr(self, "max_tokens") and self.max_tokens is not None and self.max_tokens != env_limit:
            run_max_tokens = max(1, min(dynamic_max, self.max_tokens))

        try:
            # Pick json_schema if capable, or raw json_object otherwise
            resp_format = AgentEditResponse if self.supports_schema else {"type": "json_object"}

            response = litellm.completion(
                model=self.model,
                messages=messages,
                response_format=resp_format,
                temperature=self.temperature,
                max_tokens=run_max_tokens,
                request_timeout=settings.llm_request_timeout,  # Enforce mandatory request timeout
            )

            if not response.choices:
                raise ValueError("LLM returned no choices.")

            choice = response.choices[0]
            content = choice.message.content
            finish_reason = getattr(choice, "finish_reason", None)

            # Check stop condition (Premature Cutoff)
            if finish_reason == "length" or not content:
                length_or_zero = len(content) if content else 0
                raise ValueError(
                    f"LLM generation stopped prematurely. "
                    f"finish_reason: {finish_reason}. "
                    f"content_length: {length_or_zero}"
                )

            # Extract clean JSON block if prose-wrapped
            clean_content = content.strip()
            if clean_content.startswith("```json"):
                clean_content = clean_content[7:]
                if clean_content.endswith("```"):
                    clean_content = clean_content[:-3]
            elif clean_content.startswith("```"):
                clean_content = clean_content[3:]
                if clean_content.endswith("```"):
                    clean_content = clean_content[:-3]
            clean_content = clean_content.strip()

            if not (clean_content.startswith("{") and clean_content.endswith("}")):
                start_idx = clean_content.find("{")
                end_idx = clean_content.rfind("}")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    clean_content = clean_content[start_idx : end_idx + 1]

            # Parse using Pydantic validation
            parsed_response = AgentEditResponse.model_validate_json(clean_content)

            usage = getattr(response, "usage", None)
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0

            try:
                cost = litellm.completion_cost(completion_response=response) or 0.0
            except Exception:
                cost = 0.0

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
            )

        except Exception as e:
            retryable = True
            finish_reason = None
            if isinstance(e, ValueError) and "stopped prematurely" in str(e):
                if "finish_reason: length" in str(e):
                    finish_reason = "length"
            elif "length" in str(e).lower() or getattr(e, "finish_reason", None) == "length":
                finish_reason = "length"

            # Treat BadRequestError, AuthenticationError, NotFoundError as non-retryable config errors
            if isinstance(
                e,
                (
                    litellm.BadRequestError,  # type: ignore[attr-defined]
                    litellm.AuthenticationError,  # type: ignore[attr-defined]
                    litellm.NotFoundError,  # type: ignore[attr-defined]
                ),
            ) or getattr(e, "status_code", None) in (400, 401, 403, 404):
                retryable = False

            raise LLMError(
                provider="litellm",
                model=self.model,
                detail=str(e),
                retryable=retryable,
                finish_reason=finish_reason,
            ) from e
