import pytest

from autobacktest.llm.base import AgentContext, AgentEdit
from autobacktest.llm.mock_provider import MockProvider


def test_mock_provider_identity_default() -> None:
    provider = MockProvider()
    assert provider.provider_name == "mock"
    assert len(provider.calls) == 0

    context = AgentContext(
        strategy_name="haa",
        strategy_code="def signals(): pass",
        config_yaml="universe: []",
        program_text="none",
        evaluation_report=None,
        iteration=1,
    )

    edit = provider.generate_edit(context)

    assert len(provider.calls) == 1
    assert provider.calls[0] == context
    assert edit.strategy_code == "def signals(): pass"
    assert edit.config_yaml == "universe: []"
    assert "Identity" in edit.reasoning
    assert edit.lessons_text is not None
    assert "Mock lesson recorded" in edit.lessons_text


def test_mock_provider_configured_response() -> None:
    expected_edit = AgentEdit(
        strategy_code="def new_signals(): pass",
        config_yaml="universe: [SPY]",
        reasoning="Changed signals",
        raw_response="{}",
        lessons_text="Custom mock lessons",
    )
    provider = MockProvider(response=expected_edit)
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def signals(): pass",
        config_yaml="universe: []",
        program_text="none",
        evaluation_report=None,
        iteration=2,
    )

    edit = provider.generate_edit(context)

    assert len(provider.calls) == 1
    assert edit == expected_edit


def test_mock_provider_configured_error() -> None:
    provider = MockProvider(error=ValueError("Test simulation failure"))
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def signals(): pass",
        config_yaml="universe: []",
        program_text="none",
        evaluation_report=None,
        iteration=3,
    )

    with pytest.raises(ValueError) as exc_info:
        provider.generate_edit(context)

    assert len(provider.calls) == 1
    assert str(exc_info.value) == "Test simulation failure"
