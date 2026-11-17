package main

import (
	"context"
	"fmt"
	"os"

	"dagger.io/dagger"
)

// Helper Functions

func isValidImageName(name string) bool {
	if name == "" {
		return false
	}
	for _, char := range name {
		if char == ' ' {
			return false
		}
	}
	return true
}

func checkFileExists(filePath string) bool {
	info, err := os.Stat(filePath)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir() //returns true if it exists and is a file
}

func getAppEnvironment() string {
	env := os.Getenv("APP_ENV")

	//if nothing was set, default to development
	if env == "" {
		return "development"
	}
	return env
}

func main() {
	if !checkFileExists("Dockerfile") {
		fmt.Println("Error: Dockerfile is missing.")
		return
	}

	// Create a shared context
	ctx := context.Background()

	// Run the stages of the pipeline
	if err := Build(ctx); err != nil {
		fmt.Println("Error:", err)
		panic(err)
	}
}

func Build(ctx context.Context) error {
	// Initialize Dagger client
	client, err := dagger.Connect(ctx)
	if err != nil {
		return err
	}
	defer client.Close()

	// Reads the Dockerfile and builds the container
	container := client.Host().Directory("..").DockerBuild()

	// Runs a check inside the built container to verify FastAPI loads
	_, err = container.WithExec([]string{
		"python", "-c", "import src.deployment.fastapi",
	}).Sync(ctx)

	if err != nil {
		return err
	}

	fmt.Println("Dagger succesfully built and verified the project container.")
	return nil
}
