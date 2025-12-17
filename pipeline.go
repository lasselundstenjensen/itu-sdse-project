package main

import (
	"context"
	"fmt"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

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

	// Host the entire repo root (include DVC metadata)
	repo := client.Host().Directory(".", dagger.HostDirectoryOpts{
		Include: []string{
			"cookiecutter_pipeline/itu-sdse-project-rgofv/**",
			".git/**",
			".dvc/**",
			"dvc.yaml",
			"dvc.lock",
			"**/*.dvc",
		},
	})

	// Persistent pip cache
	pipCache := client.CacheVolume("pip-cache")

	// Base container
	base := client.Container().
		From("python:3.12.2-bookworm").
		WithMountedCache("/root/.cache/pip", pipCache).
		WithEnvVariable("PIP_DISABLE_PIP_VERSION_CHECK", "1").
		WithEnvVariable("PYTHONUNBUFFERED", "1").
		WithEnvVariable("DVC_NO_ANALYTICS", "1").
		WithDirectory("/repo", repo).
		WithWorkdir("/repo/cookiecutter_pipeline/itu-sdse-project-rgofv")

	// Install git, dvc, and editable package
	base = base.WithExec([]string{
		"bash", "-lc",
		"apt-get update && apt-get install -y --no-install-recommends git && " +
			"python -m pip install --upgrade pip && " +
			"python -m pip install dvc && " +
			"python -m pip install -e .",
	})

	if _, err := base.Stdout(ctx); err != nil {
		return err
	}

	fmt.Println("Pulling data with DVC")
	base = base.WithExec([]string{
		"bash", "-lc",
		"dvc pull -v",
	})

	if _, err := base.Stdout(ctx); err != nil {
		return err
	}

	fmt.Println("Running entire pipeline")

	// Run main.py
	pipelineRun := base.WithExec([]string{
		"python", "main.py",
	})

	if _, err := pipelineRun.Stdout(ctx); err != nil {
		return err
	}

	fmt.Println("Exporting Artifacts and Data")

	// Export base_data and models
	if _, err := pipelineRun.Directory("/repo/cookiecutter_pipeline/itu-sdse-project-rgofv/base_data").Export(ctx, "base_data"); err != nil {
		return err
	}
	if _, err := pipelineRun.Directory("/repo/cookiecutter_pipeline/itu-sdse-project-rgofv/models").Export(ctx, "models"); err != nil {
		return err
	}

	// Export artifacts and mlruns
	if _, err := pipelineRun.Directory("/repo/cookiecutter_pipeline/itu-sdse-project-rgofv/models/artifacts").Export(ctx, "artifacts"); err != nil {
		return err
	}

	fmt.Println("Pipeline complete")
	return nil
}
