from mandol.auto_builder.strategy_config import STYLE_ALIASES, STYLE_STRATEGIES


def test_example_does_not_change_benchmark_strategy_aliases():
    assert STYLE_ALIASES["locomo"] == "locomo10"
    assert "locomo10" in STYLE_STRATEGIES
    assert "longmemeval" in STYLE_STRATEGIES

