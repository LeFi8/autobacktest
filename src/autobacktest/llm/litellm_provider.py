"""LiteLLM integration provider implementing structured response outputs."""

import litellm
from pydantic import BaseModel, Field

from autobacktest.llm.base import AgentContext, AgentEdit, LLMError, LLMProvider
from autobacktest.llm.prompts import build_messages


class AgentEditResponse(BaseModel):
    """Pydantic schema used for structured output parsing via LiteLLM."""

    strategy_code: str = Field(
        description="The complete new Python strategy source code."
    )
    config_yaml: str = Field(
        description="The complete new YAML parameters configuration."
    )
    reasoning: str = Field(
        description="Quantitative reasoning and explanation for changes."
    )


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

        try:
            # We use litellm completion with response_format referencing BaseModel
            response = litellm.completion(
                model=self.model,
                messages=messages,
                response_format=AgentEditResponse,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM returned an empty or invalid content response.")

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

            return AgentEdit(
                strategy_code=parsed_response.strategy_code,
                config_yaml=parsed_response.config_yaml,
                reasoning=parsed_response.reasoning,
                raw_response=content,
            )

        except Exception as e:
            raise LLMError(
                provider="litellm",
                model=self.model,
                detail=str(e),
            ) from e
