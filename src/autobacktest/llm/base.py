"""Base abstractions and domain models for the LLM driver module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    """

    strategy_name: str
    strategy_code: str
    config_yaml: str
    program_text: str
    evaluation_report: EvaluationReport | None
    iteration: int
    lessons_text: str = ""


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

    def __init__(self, provider: str, model: str, detail: str) -> None:
        """Initialize the error with details of the failing call.

        Args:
            provider: The name of the LLM provider (e.g. "openai", "mock").
            model: The name of the model that failed.
            detail: Detailed error message from the provider or library.
        """
        super().__init__(f"LLMError (provider={provider}, model={model}): {detail}")
        self.provider = provider
        self.model = model
        self.detail = detail

    def __str__(self) -> str:
        return (
            f"LLMError(provider='{self.provider}', "
            f"model='{self.model}', detail='{self.detail}')"
        )


class LLMProvider(ABC):
    """Abstract base class defining the contract for LLM drivers."""

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
