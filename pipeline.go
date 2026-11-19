// pipeline.go runs the project's pipeline in containers with Dagger.
// Run it from the project root: go run pipeline.go
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"dagger.io/dagger"
)

const baseImage = "ghcr.io/astral-sh/uv:python3.13-bookworm"

func main() {
	ctx := context.Background()
	if err := run(ctx); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// run connects to Dagger and chains the stages in order.
func run(ctx context.Context) error {
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stderr))
	if err != nil {
		return fmt.Errorf("connect to Dagger: %w", err)
	}
	defer client.Close()

	base := baseContainer(client)
	tested := unitTestStage(base)
	withData := dataStage(tested)
	//<- training stage goes here
	//<- model exporting goes here (currently being done in the deployment.ipynb)
	// check that the model was actually exported to the artifacts directory
	err = checkArtifact("models/model")
	if err != nil {
		return fmt.Errorf("pipeline failed: %w", err)
	}

	//<- model test stage goes here

	// Nothing runs until a result is requested; Sync runs all stages above.
	if _, err := withData.Sync(ctx); err != nil {
		return fmt.Errorf("pipeline failed: %w", err)
	}
	fmt.Println("Pipeline finished")

	// Here will be the pipeline part that produces the serving container:
	// (for now this just prints the image reference under which the image will be published to ghcr.io)

	// get the sha of the current commit
	sha, err := currentCommitSHA()
	if err != nil {
		return fmt.Errorf("pipeline failed: %w", err)
	}
	// construct the image reference from the sha
	imgRef, err := imageRef("Niels04", sha)
	if err != nil {
		return fmt.Errorf("pipeline failed: %w", err)
	}
	fmt.Println(imgRef) // <- replace this with publishing the container

	// Only publish as "latest" if the current branch is the main branch:
	if os.Getenv("GITHUB_REF") == "refs/heads/main" {
		imgRef, err := imageRef("Niels04", "latest")
		if err != nil {
			return fmt.Errorf("pipeline failed: %w", err)
		}
		fmt.Println(imgRef) // <- replace this with publishing the container
	}

	return nil
}

// -------- PIPELINE STEPS

// baseContainer copies the project into a Python container and installs
// exactly the packages locked in uv.lock.
func baseContainer(client *dagger.Client) *dagger.Container {
	source := client.Host().Directory(".", dagger.HostDirectoryOpts{
		Exclude: []string{".venv", "artifacts", "models/model", "mlflow-state", "data/images/raw", "data/test/*.jpg"},
	})
	return client.Container().
		From(baseImage).
		WithMountedCache("/root/.cache/uv", client.CacheVolume("uv-cache")).
		WithEnvVariable("UV_LINK_MODE", "copy").
		WithDirectory("/app", source).
		WithWorkdir("/app").
		WithExec([]string{"uv", "sync", "--frozen"})
}

// unitTestStage runs the tests in tests/ with pytest.
func unitTestStage(c *dagger.Container) *dagger.Container {
	return c.WithExec([]string{"uv", "run", "pytest"})
}

// dataStage fetches the metadata files tracked by DVC.
// The image download is added when the training script is merged.
func dataStage(c *dagger.Container) *dagger.Container {
	return c.WithExec([]string{"uv", "run", "dvc", "pull"})
}

// ------- UTILITY FUNCTIONS

// imageRef returns the name of our serving image on GHCR.
func imageRef(owner, sha string) (string, error) {
	if owner == "" || sha == "" {
		return "", errors.New("owner and sha must not be empty")
	}
	// GHCR only accepts lowercase names, so "Niels04" becomes "niels04".
	return "ghcr.io/" + strings.ToLower(owner) + "/itu-sdse-project:" + sha, nil

}

// currentCommitSHA() returns the SHA of the current git commit
func currentCommitSHA() (string, error) {
	if sha := strings.TrimSpace(os.Getenv("GITHUB_SHA")); sha != "" {
		return sha, nil
	}

	output, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		return "", fmt.Errorf("determine commit SHA: %w", err)
	}

	sha := strings.TrimSpace(string(output))
	if sha == "" {
		return "", fmt.Errorf("commit SHA is empty")
	}
	return sha, nil
}

// checkArtifact returns an error if dir does not contain an exported MLflow model.
func checkArtifact(dir string) error {
	if _, err := os.Stat(filepath.Join(dir, "MLmodel")); err != nil {
		return fmt.Errorf("no MLflow model in %s: %w", dir, err)
	}
	return nil
}
