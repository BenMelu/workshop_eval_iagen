"""Step 3 - Métrique de sécurité (Role violation).
"""

from eval.common.deepeval_model import build_deepeval_model
from deepeval.metrics import RoleViolationMetric


# TODO: Définir la fonction `role_violation_metric(threshold: float, role: str) -> RoleViolationMetric`.

def role_violation_metric(threshold: float, role: str) -> RoleViolationMetric:
    model = build_deepeval_model()
    return RoleViolationMetric(
        model=model,
        threshold=threshold,
        role=role,
    )