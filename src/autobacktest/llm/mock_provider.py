"""Mock provider implementation for local unit and integration tests."""

from autobacktest.config import settings
from autobacktest.llm.base import AgentContext, AgentEdit, LLMProvider


class MockProvider(LLMProvider):
    """Local mock implementation of LLMProvider for local unit tests."""

    def __init__(
        self,
        response: AgentEdit | None = None,
        error: Exception | None = None,
    ) -> None:
        """Initialize the mock provider with optional custom behaviors.

        Args:
            response: Pre-configured AgentEdit to return.
            error: Pre-configured Exception to raise.
        """
        self.response = response
        self.error = error
        self.calls: list[AgentContext] = []
        # Expose temperature so the orchestrator's decay schedule is exercised.
        self.temperature: float = settings.llm_temperature

    @property
    def provider_name(self) -> str:
        """Return the unique string identification of the provider."""
        return "mock"

    def generate_edit(self, context: AgentContext) -> AgentEdit:
        """Process the context using predefined mock rules and record call history.

        Args:
            context: The AgentContext to mutate.

        Returns:
            AgentEdit matching mock configuration.

        Raises:
            Exception: If an error is configured on the provider.
        """
        self.calls.append(context)

        if self.error is not None:
            raise self.error

        if self.response is not None:
            return self.response

        # Identity default: return strategy code with a mock comment to
        # represent the prompt edit and keep it valid python code,
        # unless program_text is "none" or empty.
        edited_code = context.strategy_code
        prompt_comment = (
            context.program_text.strip().replace("\n", " ")
            if context.program_text
            else ""
        )
        if prompt_comment and prompt_comment != "none":
            edited_code += f"\n# Mock edit for: {prompt_comment}\n"
            reasoning = f"Mock transformation reflecting prompt: {prompt_comment}"
        else:
            reasoning = "Identity transformation: no edits made."

        lessons_suffix = "\n- Mock lesson recorded."
        lessons_text = (
            context.lessons_text + lessons_suffix
            if context.lessons_text
            else "- Mock lesson recorded."
        )

        return AgentEdit(
            strategy_code=edited_code,
            config_yaml=context.config_yaml,
            reasoning=reasoning,
            raw_response="{}",
            lessons_text=lessons_text,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost=0.0,
        )
