package main

import (
	"context"
	"log"
	"path/filepath"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

	client, err := dagger.Connect(ctx, dagger.WithLogOutput(log.Writer()))
	if err != nil {
		panic(err)
	}
	defer client.Close()

	// Get absolute path to project root
	absPath, err := filepath.Abs("../") // relative to dagger/ folder
	if err != nil {
		panic(err)
	}

	// Mount project root
	src := client.Host().Directory(absPath)

	python := client.Container().
		From("python:3.11-slim").
		WithMountedDirectory("/app", src).
		WithWorkdir("/app/cookie_eaters").
		WithExec([]string{"pip", "install", "-r", "requirements.txt"})

	steps := []string{
		"python -m code.data.B_setup_data",
		"python -m code.data.C_preprocessing",
		"python -m code.features.D_feature_engineering",
		"python -m code.models.F_prepare_train_data",
		"python -m code.models.H_sklearn_train_and_evaluate",
		"python -m code.models.J_mlflow_model_selection",
		"python -m code.models.K_check_production_model",
		"python -m code.models.L_compare_and_register_model",
	}

	for _, step := range steps {
		log.Println("Running:", step)
		python = python.WithExec([]string{"sh", "-c", step})
	}

	_, err = python.ExitCode(ctx)
	if err != nil {
		panic(err)
	}
}
