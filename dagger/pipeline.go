package main

import (
	"context"
	"fmt"

	"dagger.io/dagger"
)

func Pipeline(ctx context.Context) error {

	// Connect to Dagger engine
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(nil))
	if err != nil {
		return err
	}
	defer client.Close()

	// Base Python container
	python := client.Container().
		From("python:3.12-bookworm").
		WithWorkdir("/app")

	// Mount your local Python project
	python = python.WithDirectory("/app", client.Host().Directory("./cookie_eaters/code"))

	// Install dependencies if you use requirements
	python = python.WithExec([]string{"pip", "install", "-r", "requirements.txt"})

	// Define pipeline Python scripts
	steps := []string{
		"01_setup_data.py",
		"02_helper_functions.py",
		"03_preprocessing.py",
		"04_feature_engineering.py",
		"05_setup_experiment.py",
		"06_load_train_data.py",
		"07_model_train.py",
		"08_evaluation.py",
		"09_Sklearning_logostoc_regress.py", // adjust actual filename
		"10_save_model.py",
		"11_mlflow_experiment.py",
		"12_production_model.py",
		"13_compare_register.py",
		"14_deploy.py",
	}

	for _, step := range steps {
		fmt.Println("▶ Running:", step)
		python = python.WithExec([]string{"python", step})
	}

	// Export artifacts (models, mlruns, etc.)
	_, err = python.Directory("/app").Export(ctx, "./output")
	if err != nil {
		return err
	}

	fmt.Println("Pipeline completed")
	return nil
}
