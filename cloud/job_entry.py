"""Entry point for the Nebius Serverless Job: clone the repo, install the pinned stack, run cloud_eval.py.

The job runs `python -c <one-liner>` that downloads and executes this file, which avoids any shell quoting
in the job's command-line arguments.
"""
import subprocess
import sys

REPO = "https://github.com/VishwaVenkatesh/nemotron-reward-engineer.git"


def sh(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


sh(["git", "clone", "--depth", "1", REPO, "/app"])
sh([sys.executable, "-m", "pip", "install", "-q", "-r", "/app/cloud/requirements-cloud.txt"])
sh([sys.executable, "/app/cloud_eval.py", "--episodes", "64", "--horizon", "500"])
