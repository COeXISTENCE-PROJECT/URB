import math
import os
from pathlib import Path

import pytest

from leaderboard.generate_leaderboard import collect_experiments


REPO_ROOT = Path(__file__).resolve().parents[1]
NEW_RESULTS = [
    REPO_ROOT / path
    for path in os.environ.get("URB_NEW_RESULTS", "").splitlines()
    if path.strip()
]


@pytest.mark.parametrize("result_dir", NEW_RESULTS, ids=lambda path: path.name)
def test_new_result_is_indexable(result_dir, tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / result_dir.name).symlink_to(result_dir, target_is_directory=True)

    indexed = collect_experiments(results_dir)

    assert len(indexed) == 1, "result needs a valid exp_config.json and complete metrics"
    result = indexed[0]
    assert result["exp_id"] == result_dir.name
    assert result["network"]
    assert result["script"]
    assert math.isfinite(float(result["metrics"]["t_test"]))
    assert math.isfinite(float(result["metrics"]["t_CAV"]))

