"""Step 3 - Métrique outillage (Tool correctness).
"""
from eval.common.deepeval_model import build_deepeval_model
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import ToolCallParams


# TODO: Définir la fonction `tool_correctness_metric(threshold: float, should_exact_match: bool = True,
#       should_consider_ordering: bool = False) -> ToolCorrectnessMetric`.

def tool_correctness_metric(threshold: float, should_exact_match: bool, should_consider_ordering: bool) -> ToolCorrectnessMetric:
    model = build_deepeval_model()
    return ToolCorrectnessMetric(
        model=model,
        threshold=threshold,
        should_exact_match=should_exact_match,
        should_consider_ordering=should_consider_ordering,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        async_mode=False,
    )