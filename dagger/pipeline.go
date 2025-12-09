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
	python = python.WithDirectory("/app", client.Host().Directory("../cookie_eaters"))

	// Install dependencies if you use requirements
	python = python.WithExec([]string{"pip", "install", "-r", "requirements.txt"})

	// Define pipeline Python scripts
	steps := []string{
		"./code/data/01_setup_data.py",
		"./code/data/02_helper_functions.py",
		"./code/data/03_preprocessing.py",
		"./code/features/04_feature_engineering.py",
		"./code/models/05_setup_experiment.py",
		"./code/models/06_load_train_data.py",
		"./code/models/07_model_train.py",
		"./code/models/08_evaluation.py",
		"./code/models/09_Sklearin_logistic_regression.py",
		"./code/models/10_save_model.py",
		"./code/models/11_mlflow_experiment.py",
		"./code/models/12_production_model.py",
		"./code/models/13_compare_register.py",
		"./code/models/14_deploy.py",
	}

	for _, step := range steps {
		fmt.Println("Running:", step)
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
