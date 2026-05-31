"""Base abstractions and domain models for the LLM driver module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from autobacktest.evaluator.report import EvaluationReport


@dataclass(frozen=True)
class AgentContext:
    """Immutable context provided to the LLM agent for strategy generation.

    Attributes:
        strategy_name: Name of the target strategy.
        strategy_code: Current python source of the strategy file.
        config_yaml: Current raw YAML configuration content of the strategy.
        program_text: Full objective and constraints markdown text.
        evaluation_report: Evaluation report from the previous iteration, if any.
        iteration: The 1-indexed counter of the current optimization loop.
        last_attempt: Dict describing the most recent failed iteration, if any.
            Keys vary by failure stage: stage, error_code, detail, rejection_reason,
            failed_gate, candidate_strategy_code, candidate_config_yaml,
            candidate_metrics.
    """

    strategy_name: str
    strategy_code: str
    config_yaml: str
    program_text: str
    evaluation_report: EvaluationReport | None
    iteration: int
    lessons_text: str = ""
    n_historical_configs: int = 0
    last_attempt: dict[str, Any] | None = None


@dataclass(frozen=True)
class AgentEdit:
    """Immutable structured edit returned by the LLM driver.

    Attributes:
        strategy_code: Complete new python source for the strategy file.
        config_yaml: Complete new YAML configuration content.
        reasoning: Text justification of modifications made.
        raw_response: Raw response string from the provider for audit logging.
        lessons_text: Updated lessons markdown text, or None to leave it unchanged.
    """

    strategy_code: str
    config_yaml: str
    reasoning: str
    raw_response: str
    lessons_text: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0


class LLMError(Exception):
    """Domain exception raised when an LLM provider fails."""

    def __init__(
        self,
        provider: str,
        model: str,
        detail: str,
        retryable: bool = True,
        finish_reason: str | None = None,
    ) -> None:
        """Initialize the error with details of the failing call.

        Args:
            provider: The name of the LLM provider (e.g. "openai", "mock").
            model: The name of the model that failed.
            detail: Detailed error message from the provider or library.
            retryable: Whether the error is transient and can be retried.
            finish_reason: The token completion stop condition if truncated (e.g. "length").
        """
        super().__init__(f"LLMError (provider={provider}, model={model}): {detail}")
        self.provider = provider
        self.model = model
        self.detail = detail
        self.retryable = retryable
        self.finish_reason = finish_reason

    def __str__(self) -> str:
        return (
            f"LLMError(provider='{self.provider}', model='{self.model}', "
            f"detail='{self.detail}', retryable={self.retryable}, "
            f"finish_reason='{self.finish_reason}')"
        )


class LLMProvider(ABC):
    """Abstract base class defining the contract for LLM drivers."""

    temperature: float

    @abstractmethod
    def generate_edit(self, context: AgentContext) -> AgentEdit:
        """Consume context and generate strategy modifications.

        Args:
            context: Current state and objective.

        Returns:
            AgentEdit containing the proposed updates.

        Raises:
            LLMError: If LLM service, parser, or network fails.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique string identification of the provider."""
        pass
