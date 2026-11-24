package main

import (
	"context"
	"fmt"
	"os"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stderr))
	if err != nil {
		fmt.Fprintln(os.Stderr, "could not connect to Dagger:", err)
		os.Exit(1)
	}
	defer client.Close()

	steps := []string{"get data", "prepare data", "train model", "save model"}

	for _, step := range steps {
		out, err := runStep(ctx, client, step)
		if err != nil {
			fmt.Fprintln(os.Stderr, "step failed:", step, err)
			os.Exit(1)
		}
		fmt.Print(out)
	}
}

// runStep is a placeholder. Later it will run the real commands.
func runStep(ctx context.Context, client *dagger.Client, name string) (string, error) {
	return client.Container().
		From("python:3.13-slim").
		WithExec([]string{"echo", "PLACEHOLDER: " + name}).
		Stdout(ctx)
}
