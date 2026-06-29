import json

from autobacktest.reports.generator import compile_failure_summary


def test_compile_failure_summary_counts_llm_error_details(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    event = {
        "iteration": 1,
        "candidates": [
            {
                "llm_error": True,
                "detail": "LLMError(provider='litellm', model='x', detail='finish_reason: length')",
                "finish_reason": "length",
            },
            {"llm_error": True, "detail": "Malformed JSON returned by model"},
        ],
    }
    (run_dir / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")

    summary = compile_failure_summary(run_dir)

    assert summary["LLM Error"] == 2
    assert summary["LLM Error Details"] == {
        "finish_reason=length": 1,
        "Malformed JSON returned by model": 1,
    }
