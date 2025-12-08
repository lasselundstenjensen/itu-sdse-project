package main

import (
	"context"
	"fmt"

	"dagger.io/dagger"
)

func main() {
	// Create a shared context
	ctx := context.Background()

	// Run the stages of the pipeline
	if err := Build(ctx); err != nil {
		fmt.Println("Error:", err)
		panic(err)
	}
}

func Build(ctx context.Context) error {
	client, err := dagger.Connect(ctx)
	if err != nil {
		return err
	}
	defer client.Close()

	// 1. Setup Resources (Host files, caches)
	src := client.Host().Directory(".")
	pipCache := create_cache(client, "pip_cache")

	// 2. Setup Container Base
	python := create_container(client, "python:3.12")
	python = set_directory(python, "/app", src)
	python = set_workdir(python, "/app")
	python = mount_cache(python, pipCache, "/root/.cache/pip")

	// 3. Prepare Environment
	python = python.WithExec([]string{"mkdir", "-p", "new_customers_classifier"})
	python = python.WithExec([]string{"touch", "new_customers_classifier/__init__.py"})

	// 4. Install Dependencies
	python = install_requirements(python)

	// 5. Setup DVC (Git init + Pull)
	python = pull_data(python)

	// 6. Run ML Pipeline Stages
	python = run_script(python, "new_customers_classifier/preprocessing.py", "/app/data/raw/raw_data.csv")
	python = run_script(python, "new_customers_classifier/model_dev.py", "/app/data/processed/train_data_gold.csv")
	python = run_script(python, "new_customers_classifier/model_selection.py")
	python = run_script(python, "new_customers_classifier/deploy.py")

	// 7. Export Artifacts
	if err := export_artifacts(python, ctx, "models"); err != nil {
		return err
	}

	return nil
}

func create_container(client *dagger.Client, image string) *dagger.Container {
	return client.Container().From(image)
}

func set_directory(container *dagger.Container, path string, src *dagger.Directory) *dagger.Container {
	return container.WithDirectory(path, src)
}

func set_workdir(container *dagger.Container, workdir string) *dagger.Container {
	return container.WithWorkdir(workdir)
}

func create_cache(client *dagger.Client, name string) *dagger.CacheVolume {
	return client.CacheVolume(name)
}

func mount_cache(container *dagger.Container, cache *dagger.CacheVolume, mountPath string) *dagger.Container {
	return container.WithMountedCache(mountPath, cache)
}

func install_requirements(container *dagger.Container) *dagger.Container {
	return container.WithExec([]string{"pip", "install", "."})
}

func pull_data(container *dagger.Container) *dagger.Container {
	container = container.WithExec([]string{"git", "init"})
	container = container.WithExec([]string{"dvc", "update", "data/raw/raw_data.csv.dvc"})

	return container
}

func run_script(container *dagger.Container, scriptPath string, args ...string) *dagger.Container {
	cmd := append([]string{"python", scriptPath}, args...)
	return container.WithExec(cmd)
}

func export_artifacts(container *dagger.Container, ctx context.Context, exportPath string) error {
	_, err := container.
		Directory("models").
		Export(ctx, exportPath)
	return err
}
