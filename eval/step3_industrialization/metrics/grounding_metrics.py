"""Step 3 - Métrique de grounding (Faithfulness).
"""

from eval.common.deepeval_model import build_deepeval_model
from deepeval.metrics import FaithfulnessMetric


# TODO: Définir la fonction
#       `faithfulness_metric(threshold: float, truths_extraction_limit: int = 4) -> FaithfulnessMetric`.

def faithfulness_metric(threshold: float, truths_extraction_limit: int) -> FaithfulnessMetric:
    model = build_deepeval_model()
    return FaithfulnessMetric(
        model=model,
        threshold=threshold,
        truths_extraction_limit=truths_extraction_limit,
        penalize_ambiguous_claims=True,
    )