import os
import pytest
import shutil
import subprocess

from pathlib import Path

SCRIPTS_DIR = Path("scripts")
baseline_scripts = [
    SCRIPTS_DIR / "baselines.py",
    SCRIPTS_DIR / "baselines_clusters.py",
]

BASELINES_DIR = Path("baseline_models")
baseline_names = list(BASELINES_DIR.rglob("*.py"))
baseline_names = [name for name in baseline_names if name.name not in ["__init__.py", "base.py", "registry.py"]]

@pytest.fixture(scope="session", autouse=True)
def check_sumo_installed():
    sumo_executable = shutil.which("sumo")
    if sumo_executable is None:
        pytest.exit("[SUMO ERROR] SUMO is not installed or not in PATH.")
    else:
        try:
            result = subprocess.run(
                ["sumo", "--version"], capture_output=True, text=True, check=True
            )
            print(f"[DEBUG] SUMO version: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            pytest.exit(f"[SUMO ERROR] Failed to get SUMO version: {e.stderr}")

@pytest.mark.parametrize("script_path", baseline_scripts)
@pytest.mark.parametrize("baseline", baseline_names)
def test_python_script_execution(script_path, baseline):
    try:
        script_filename = script_path.name
        print(script_filename)
        baseline_name = baseline.name.split(".")[0]
        print(baseline_name)
        result = subprocess.run(
            ["python", script_filename,
             "--id", f"test_{script_filename}_{baseline_name}",
             "--alg-conf", "test",
             "--env-conf", "test",
             "--task-conf", "test",
             "--net", "saint_arnoult",
             "--model", baseline_name],
            capture_output=True, text=True, check=True, cwd=script_path.parent
        )
        print(f"[DEBUG] Successfully executed baseline {baseline_name} with {script_path}")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"[FAIL] Baseline {baseline_name} failed: {e.stderr}")