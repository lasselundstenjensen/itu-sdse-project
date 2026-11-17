import atexit
import subprocess
import sys
import os

export_dir = "models/model"
model_path = export_dir

upstream_port = 5002
upstream_host = "127.0.0.1"

def start_scoring_server():
    """launches MLflow's model server as a background process 
    and turns it off once our code ends"""

    env = os.environ.copy()
    interpreter_bin = os.path.dirname(os.path.abspath(sys.executable))
    env["PATH"] = os.pathsep.join([interpreter_bin, env.get("PATH", "")])
    proc = subprocess.Popen([
        sys.executable, "-m", "mlflow", "models", "serve",
        "-m", model_path,
        "-p", str(upstream_port),
        "--workers", "1",
        "--host", upstream_host,
        "--no-conda",
    ], env=env)
    atexit.register(lambda: proc.poll() is None and proc.terminate())
    return proc



