//PLACEHOLDER FUNCTION UNTIL WE SORT THE NOTEBOOK FILES INTO PYTHON ONES

package main

import (
	"dagger/pipeline/internal/dagger"
)

type Pipeline struct{}

// Env builds the Python environment with the project's dependencies installed
func (m *Pipeline) Env(
	// +defaultPath="/"
	// +ignore=[".git", ".venv", "**/__pycache__", "dagger"]
	source *dagger.Directory,
) *dagger.Container {
	return dag.Container().
		From("python:3.10-slim").
		WithMountedCache("/root/.cache/pip", dag.CacheVolume("pip-cache")). // reuse downloaded wheels
		// CPU-only torch wheels: much smaller than the default CUDA build
		WithEnvVariable("PIP_EXTRA_INDEX_URL", "https://download.pytorch.org/whl/cpu").
		WithWorkdir("/app").
		// Copy ONLY requirements.txt first, so this install step stays cached
		// until requirements.txt itself changes (code edits won't trigger a reinstall).
		WithFile("/app/requirements.txt", source.File("requirements.txt")).
		WithExec([]string{"sh", "-c", "grep -v '^-e' requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt"}).
		// Now bring in the rest of the code; only this cheap step reruns on code changes
		WithDirectory("/app", source).
		WithExec([]string{"pip", "install", "--no-deps", "-e", "."})
}

// Train runs the training script and returns the artifacts directory
// (vial_cnn_model.pth, model_results.json, drift/class/image stats JSONs)
func (m *Pipeline) Train(
	// +defaultPath="/"
	// +ignore=[".git", ".venv", "**/__pycache__", "dagger", "artifacts", "mlflow-state", "models"]
	source *dagger.Directory,
) *dagger.Directory {
	return m.trained(source).Directory("/app/artifacts")
}

// Deploy trains the model, then runs deployment.py: picks the best MLflow run,
// promotes it to Production, exports it, serves it locally and smoke-tests
// the endpoint + drift check. Returns the exported servable model (models/model).
func (m *Pipeline) Deploy(
	// +defaultPath="/"
	// +ignore=[".git", ".venv", "**/__pycache__", "dagger", "artifacts", "mlflow-state", "models"]
	source *dagger.Directory,
) *dagger.Directory {
	return m.trained(source).
		WithExec([]string{"python", "./source/deployment.py"}).
		Directory("/app/models/model")
}

// trained is the environment after training.py has run, so artifacts/,
// mlflow-state/ and data/ exist for later steps. Not exposed to the CLI.
// Train and Deploy share this step, so with the same source the training
// run is cached and not repeated.
func (m *Pipeline) trained(source *dagger.Directory) *dagger.Container {
	return m.Env(source).
		WithExec([]string{"python", "./source/training.py"})
}
