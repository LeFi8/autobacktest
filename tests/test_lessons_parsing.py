from autobacktest.llm.prompts import filter_lessons, parse_lessons


def test_parse_lessons_extracts_metadata():
    lessons_text = """
### Iteration 15 (Smoke test failure)
- **Type:** BUG
- Error: "Unalignable boolean Series"
- Lesson: Keep indices aligned.

### Iteration 16 (Diversity failure)
- **Type:** DIVERSITY
- Error: High returns correlation.
- Lesson: Adopt a different universe.

### General guidelines
- **Type:** STRUCTURAL
- Hysteresis prevents whipsawing.
"""
    lessons = parse_lessons(lessons_text)
    assert len(lessons) == 3
    assert lessons[0]["title"] == "Iteration 15 (Smoke test failure)"
    assert lessons[0]["type"] == "BUG"
    assert "Unalignable boolean Series" in lessons[0]["body"]

    assert lessons[1]["title"] == "Iteration 16 (Diversity failure)"
    assert lessons[1]["type"] == "DIVERSITY"

    assert lessons[2]["title"] == "General guidelines"
    assert lessons[2]["type"] == "STRUCTURAL"


def test_filter_lessons_by_context():
    lessons_text = """
### Iteration 15 (Smoke test failure)
- **Type:** BUG
- Lesson: Fix AST.

### Iteration 16 (Diversity failure)
- **Type:** DIVERSITY
- Lesson: Alter configuration universe.

### General guidelines
- **Type:** STRUCTURAL
- Simple SMA logic works.
"""
    # 1. Test filtering for validation failures (BUG context)
    filtered_bug = filter_lessons(lessons_text, "validation")
    assert "Smoke test failure" in filtered_bug
    assert "General guidelines" in filtered_bug
    assert "Diversity failure" not in filtered_bug

    # 2. Test filtering for diversity failures (DIVERSITY context)
    filtered_div = filter_lessons(lessons_text, "diversity_config")
    assert "Diversity failure" in filtered_div
    assert "General guidelines" in filtered_div
    assert "Smoke test failure" not in filtered_div

    # 3. Test filtering with no target context
    filtered_none = filter_lessons(lessons_text, None)
    assert "Smoke test failure" in filtered_none
    assert "Diversity failure" in filtered_none
    assert "General guidelines" in filtered_none


def test_build_messages_filters_lessons():
    from autobacktest.llm.base import AgentContext
    from autobacktest.llm.prompts import build_messages

    lessons_text = """
### Iteration 15 (Smoke test failure)
- **Type:** BUG
- Lesson: Fix AST.

### Iteration 16 (Diversity failure)
- **Type:** DIVERSITY
- Lesson: Alter configuration universe.

### General guidelines
- **Type:** STRUCTURAL
- Simple SMA logic works.
"""

    context = AgentContext(
        strategy_name="toy",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe:\n  - SPY\nbenchmark: SPY",
        program_text="Maximize Sharpe.",
        evaluation_report=None,
        iteration=2,
        lessons_text=lessons_text,
        last_attempt={"stage": "validation"},  # Should trigger BUG context filtering!
    )

    messages = build_messages(context)
    user_msg = messages[1]["content"]

    assert "Smoke test failure" in user_msg
    assert "General guidelines" in user_msg
    assert "Diversity failure" not in user_msg
