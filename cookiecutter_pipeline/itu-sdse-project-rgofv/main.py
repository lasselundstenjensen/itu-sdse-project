# main.py at repo root
from mlops_pipeline.cli.app import app  # import the Typer app

if __name__ == "__main__":
    # Programmatically invoke the CLI command "run-all"
    app(["run-all"])
