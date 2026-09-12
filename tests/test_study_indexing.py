import os
from pathlib import Path

import pytest

from leaderboard.generate_leaderboard import collect_studies


REPO_ROOT = Path(__file__).resolve().parents[1]
NEW_STUDIES = [
    REPO_ROOT / path
    for path in os.environ.get("URB_NEW_STUDIES", "").splitlines()
    if path.strip()
]


@pytest.mark.parametrize("study_dir", NEW_STUDIES, ids=lambda path: path.name)
def test_new_study_is_indexable(study_dir, tmp_path):
    studies_dir = tmp_path / "studies"
    studies_dir.mkdir()
    (studies_dir / study_dir.name).symlink_to(study_dir, target_is_directory=True)

    indexed = collect_studies(
        studies_dir,
        REPO_ROOT,
        "https://github.com/COeXISTENCE-PROJECT/URB/tree/main",
        {},
    )

    assert len(indexed) == 1, "study is missing required files or experiment evidence"
    study = indexed[0]
    assert study["slug"] == study_dir.name
    assert study["title"]
    assert study["hypothesis"] and study["hypothesis"] != "See the study proposal."
    assert study["status"] in {"supported", "falsified", "inconclusive"}

