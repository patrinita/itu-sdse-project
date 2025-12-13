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
		"./code/data/B_setup_data.py",
		"./code/data/C_preprocessing.py",
		"./code/features/D_feature_engineering.py",
		"./code/models/E_setup_experiment.py",
		"./code/models/F_load_train_data.py",
		"./code/model/G_XGBoost_train_annd_evaluate.py",
		"./code/models/H_model_training.py",
		"./code/models/I_save_artifacts.py",
		"./code/models/J_mlflow_model_selection.py",
		"./code/models/K_check_production_model.py",
		"./code/models/L_compare_and_register.py",
		"./code/models/M_model_staging.py",
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
