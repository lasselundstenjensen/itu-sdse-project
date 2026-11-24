package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

	// checks if input equals expected ()

	stage, err := parseStage(os.Args)
	if err != nil {
		fmt.Println("Error:", err)
		os.Exit(1)
	}

	// opens connection to dagger engine and shows dagger logs in terminal
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stderr))
	if err != nil {
		panic(err)
	}
	defer client.Close()

	switch stage {
	case "test":
		err = Test(ctx, client)
	case "train":
		err = Train(ctx, client)
	case "publish":
		err = Publish(ctx, client, os.Getenv("IMAGE_TAG"))
	default:
		err = fmt.Errorf("unknown stage")
	}

	if err != nil {
		fmt.Println("Error:", err)
		os.Exit(1)
	}

}

// parseStage checks the command-line arguments and returns the stage name.
func parseStage(args []string) (string, error) {
	if len(args) < 2 {
		return "", fmt.Errorf("usage: go run . [test/train]")
	}
	if args[1] != "test" && args[1] != "train" {
		return "", fmt.Errorf("unknown stage: %s", args[1])
	}
	return args[1], nil
}

// take the client and return a container
func base(client *dagger.Client) *dagger.Container {
	src := client.Host().Directory("..", dagger.HostDirectoryOpts{
		//keep out th images, the .venv and only copies what we need
		Include: []string{
			"vial_classifier",
			"tests",
			"pyproject.toml",
			"uv.lock",
			"README.md",
			".dvc/config",
			// the metadata CSVs live in the DVC cache committed to git (there is no DVC remote)
			".dvc/cache",
			"data/**/*.dvc",
		},
	})

	//Starts a new container from premade image and install the depencies from uv file
	return client.Container().
		From("ghcr.io/astral-sh/uv:python3.13-bookworm-slim").
		WithDirectory("/app", src).
		WithWorkdir("/app").
		WithExec([]string{"uv", "sync", "--frozen"}).
		// .git is not copied into the container, so tell DVC to work without git.
		// --local writes .dvc/config.local inside the container only.
		WithExec([]string{"uv", "run", "dvc", "config", "--local", "core.no_scm", "true"})
}

// Runs all test_ files
func Test(ctx context.Context, client *dagger.Client) error {
	_, err := base(client).
		WithExec([]string{"uv", "run", "pytest"}).
		Sync(ctx) // This is where it actually runs
	return err
}

func Train(ctx context.Context, client *dagger.Client) error {
	const stateDir = "../.dagger-mlflow-state"
	if err := os.MkdirAll(stateDir, 0o755); err != nil {
		return err
	}

	trained := base(client).
		WithDirectory("mlflow-state", client.Host().Directory(stateDir)).
		WithExec([]string{"uv", "run", "python", "-m", "vial_classifier.pipeline", "train"}).
		WithExec([]string{"uv", "run", "python", "-m", "vial_classifier.pipeline", "export"})

	_, err := trained.Directory("models/model").Export(ctx, "../models/model")
	if err != nil {
		return err
	}

	_, err = trained.Directory("artifacts").Export(ctx, "../artifacts")
	if err != nil {
		return err
	}

	_, err = trained.Directory("mlflow-state").Export(ctx, stateDir)
	return err
}

// Builds the serving image from the trained model and pushes it to GHCR.
func Publish(ctx context.Context, client *dagger.Client, tag string) error {
	owner := strings.ToLower(os.Getenv("GITHUB_REPOSITORY_OWNER"))
	token := client.SetSecret("ghcr-token", os.Getenv("GITHUB_TOKEN"))

	src := client.Host().Directory("..", dagger.HostDirectoryOpts{
		Include: []string{
			"vial_classifier",
			"pyproject.toml",
			"README.md",
			"docker",
			"models/model",
			"artifacts/drift_baseline_stats.json",
		},
	})

	// runs a docker build on the seleccted files, creates an image
	image := src.DockerBuild(dagger.DirectoryDockerBuildOpts{
		Dockerfile: "docker/Dockerfile.serve",
	})

	_, err := image.
		WithExec([]string{"python", "-c", "import vial_classifier.serving.app"}).
		Sync(ctx)
	if err != nil {
		return err
	}

	address := "ghcr.io/" + owner + "/model-serve:" + tag
	published, err := image.
		WithRegistryAuth("ghcr.io", owner, token).
		Publish(ctx, address)
	if err != nil {
		return err
	}

	fmt.Println("Published", published)
	return nil
}
