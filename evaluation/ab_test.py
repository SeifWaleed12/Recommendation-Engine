"""Simple offline A/B simulator for recommendation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ABTestResult:
    control_mean: float
    treatment_mean: float
    lift_percent: float


def simulate_ab_test(control_scores: list[float], treatment_scores: list[float]) -> ABTestResult:
    control = float(np.mean(control_scores)) if control_scores else 0.0
    treatment = float(np.mean(treatment_scores)) if treatment_scores else 0.0
    lift = ((treatment - control) / control * 100.0) if control else 0.0
    return ABTestResult(control, treatment, lift)


if __name__ == "__main__":
    result = simulate_ab_test([0.04, 0.05, 0.06], [0.06, 0.07, 0.08])
    print(result)
