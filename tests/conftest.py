"""Point every test at a throwaway project root before rankops is imported."""
import os
import shutil
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
_tmp = Path(tempfile.mkdtemp(prefix="rankops-test-"))
_params = yaml.safe_load((REPO / "params.yaml").read_text())
_params["data"]["jobs_per_period"] = 40          # small and fast
_params["train"]["lgbm"]["n_estimators"] = 60
(_tmp / "params.yaml").write_text(yaml.safe_dump(_params))

os.environ["RANKOPS_ROOT"] = str(_tmp)
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_tmp / 'mlflow.db'}"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_tmp, ignore_errors=True)
