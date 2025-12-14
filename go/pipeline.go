package main

import (
	"fmt"
	"os"
	"os/exec"
)

func runStep(name string, command string, args ...string) {
	fmt.Println("Running:", name)

	cmd := exec.Command(command, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		fmt.Println("Failed:", name)
		os.Exit(1)
	}

	fmt.Println("Completed:", name)
}

func main() {
	runStep(
		"Install Python dependencies",
		"python",
		"-m", "pip", "install", "-r", "requirements.txt",
	)

	runStep("Data ingestion", "python", "src/data_ingestion.py")
	runStep("Feature engineering", "python", "src/data_features.py")
	runStep("Model training", "python", "src/model_training.py")
	runStep("Model evaluation", "python", "src/model_evaluation.py")

	fmt.Println("Pipeline finished successfully")
}
