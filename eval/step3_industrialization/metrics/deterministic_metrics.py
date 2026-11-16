"""Step 3 - Métrique déterministe (Json correctness).
"""

from pydantic import BaseModel
from eval.common.deepeval_model import build_deepeval_model
from deepeval.metrics import JsonCorrectnessMetric


# TODO: Définir le modèle Pydantic `ITActionPlan` (déduit des `actual_output` du dataset)
#       puis la fonction `json_correctness_metric(threshold: float) -> JsonCorrectnessMetric`.
class ItActionPlan(BaseModel):
    ticket_id: str
    status: str
    next_step: str

def json_correctness_metric(threshold: float) -> JsonCorrectnessMetric:
    """Métrique GEval pour évaluer la correction et l'opérationnalité de la réponse IT."""
    model = build_deepeval_model()

    return JsonCorrectnessMetric(
        model=model,
        threshold=threshold,
        expected_schema=ItActionPlan,
        strict_mode=True,
        async_mode=False
    )