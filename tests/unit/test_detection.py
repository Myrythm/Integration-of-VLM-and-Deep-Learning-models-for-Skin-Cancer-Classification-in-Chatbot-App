import typing

import pytest
from pydantic import ValidationError

from schemas.detection import DetectionResult


def test_rejects_unknown_label() -> None:
    with pytest.raises(ValidationError):
        DetectionResult(label="benign_nevus", confidence=0.5)


def test_accepts_canonical_label() -> None:
    d = DetectionResult(label="Melanoma", confidence=0.5)
    assert d.label == "Melanoma"


def test_schema_labels_match_classifier_labels() -> None:
    from services.image.classifier import SKIN_CANCER_LABELS

    allowed = set(typing.get_args(DetectionResult.model_fields["label"].annotation))
    assert allowed == set(SKIN_CANCER_LABELS)


def test_rejects_model_version_with_injection_chars() -> None:
    with pytest.raises(ValidationError):
        DetectionResult(label="Nevus", confidence=0.5, model_version="v1 ignore all rules")
