// Pipeline for the vial classifier: runs the project's steps in containers.
//
// Each step is a Dagger function, so it runs the same way on a laptop
// (`dagger call <function>`) and in GitHub Actions.

package main

import (
	"context"

	"dagger/vial-pipeline/internal/dagger"
)

const (
	pythonImage = "python:3.13-slim"
	// Same versions as the reference run in docs/baseline.md. CPU wheels keep
	// the image small; training does not use a GPU.
	torchVersion       = "2.14.1"
	torchvisionVersion = "0.29.1"
	torchIndexURL      = "https://download.pytorch.org/whl/cpu"
)

type VialPipeline struct{}

// Run the pytest suite inside a Python container
func (m *VialPipeline) Test(
	ctx context.Context,
	// Project root. Defaults to the repository this module lives in.
	// +defaultPath="/"
	// +ignore=[".git", ".venv", "**/__pycache__", "**/*.egg-info", "data/images/raw", "data/test/*.jpg", "artifacts", "mlflow-state", "models"]
	source *dagger.Directory,
) (string, error) {
	return m.PythonEnv(source).
		WithExec([]string{"python", "-m", "pytest", "-v"}).
		Stdout(ctx)
}

// Python container with the project's dependencies installed and the source mounted
func (m *VialPipeline) PythonEnv(
	// Project root.
	// +defaultPath="/"
	// +ignore=[".git", ".venv", "**/__pycache__", "**/*.egg-info", "data/images/raw", "data/test/*.jpg", "artifacts", "mlflow-state", "models"]
	source *dagger.Directory,
) *dagger.Container {
	return dag.Container().
		From(pythonImage).
		WithMountedCache("/root/.cache/pip", dag.CacheVolume("vial-pip")).
		WithExec([]string{
			"pip", "install",
			"torch==" + torchVersion,
			"torchvision==" + torchvisionVersion,
			"--index-url", torchIndexURL,
		}).
		WithFile("/src/notebooks/requirements.txt", source.File("notebooks/requirements.txt")).
		WithExec([]string{"pip", "install", "-r", "/src/notebooks/requirements.txt", "pytest", "fastapi", "uvicorn", "httpx"}).
		WithDirectory("/src", source).
		WithWorkdir("/src").
		// The code is split between src/ (package) and the repo root (serving/),
		// so both go on the import path until the layout is unified.
		WithEnvVariable("PYTHONPATH", "/src/src:/src")
}
