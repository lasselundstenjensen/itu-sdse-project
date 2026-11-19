// pipeline.go runs the project's pipeline in containers with Dagger.
// Run it from the project root: go run pipeline.go
package main

import (
	"context"
	"fmt"
	"os"

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

	// Nothing runs until a result is requested; Sync runs all stages above.
	if _, err := withData.Sync(ctx); err != nil {
		return fmt.Errorf("pipeline failed: %w", err)
	}
	fmt.Println("Pipeline finished")
	return nil
}

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
