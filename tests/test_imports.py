import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_imports(request):
    dummy = request.config.getoption("--dummy")
    if dummy:
        pytest.skip("Dummy")

    from software_defect_prediction.models.model import DefectClassifier

    assert DefectClassifier is not None
