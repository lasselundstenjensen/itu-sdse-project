package main

import (
	"context"
	"fmt"
	"os"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

	// checks if input equals expected ()

	if len(os.Args) < 2 {
		fmt.Println("usage: go run . [test/train]")
		os.Exit(1)

	}

	// save the stage name
	stage := os.Args[1]

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
	default:
		err = fmt.Errorf("unknown stage")
	}

	if err != nil {
		fmt.Println("Error:", err)
		os.Exit(1)
	}

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
			"data/**/*.dvc",
		},
	})

	//Starts a new container from premade image and install the depencies from uv file
	return client.Container().
		From("ghcr.io/astral-sh/uv:python3.13-bookworm-slim").
		WithDirectory("/app", src).
		WithWorkdir("/app").
		WithExec([]string{"uv", "sync", "--frozen"})
}

// Runs all test_ files
func Test(ctx context.Context, client *dagger.Client) error {
	_, err := base(client).
		WithExec([]string{"uv", "run", "pytest"}).
		Sync(ctx) // This is where it actually runs
	return err
}

func Train(ctx context.Context, client *dagger.Client) error {
	trained := base(client).
		WithExec([]string{"uv", "run", "python", "-m", "vial_classifier.pipeline", "data"}).
		WithExec([]string{"uv", "run", "python", "-m", "vial_classifier.pipeline", "train"}).
		WithExec([]string{"uv", "run", "python", "-m", "vial_classifier.pipeline", "export"})

	_, err := trained.Directory("models/model").Export(ctx, "../models/model")
	if err != nil {
		return err
	}

	_, err = trained.Directory("artifacts").Export(ctx, "../artifacts")
	return err
}
