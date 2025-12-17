package main

import (
	"context"

	"dagger.io/dagger"
)

func main() {
    ctx := context.Background()

    if err := Build(ctx); err != nil {
        panic(err)
    }
}
func Build(ctx context.Context) error {
    client, err := dagger.Connect(ctx)
    if err != nil {
	return err
}
    defer client.Close()

    python := client.Container().From("python:3.10").
        WithDirectory("/mlops_project", client.Host().Directory(".")).
	WithWorkdir("/mlops_project").
        WithExec([]string{"python", "--version"}).
        WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
        WithExec([]string{"mkdir","-p","output"}).
    	WithExec([]string{"python", "mlops_project/scripts/main.py"}).
        WithExec([]string{"ls", "-lah", "output"}) 

    _, err = python.
		Directory("output").
		Export(ctx, "output")
	if err != nil {
		return err
	}

    return nil
}

