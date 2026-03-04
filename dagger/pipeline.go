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

	//Get absolute path to project root
	absPath, err := filepath.Abs("../") //from dagger folder we directs to the project root
	if err != nil {
		panic(err)
	}

	//Mount project root
	src := client.Host().Directory(absPath) //take the directory from my computer and make it available to the container

	python := client.Container().
		From("python:3.11-slim").
		WithMountedDirectory("/app", src). //this is where the container stores the directory
		WithWorkdir("/app").
		WithExec([]string{
			"pip", "install",
			"--default-timeout=1000",
			"--no-cache-dir",
			"-r", "cookie_eaters/requirements.txt", //we install all necessary libaries listed in the requirements.txt
		}).
		WithExec([]string{"pip", "install", "dvc"}). //we instal dvc
		WithExec([]string{"dvc", "pull"}).           //we dvc pull the data from the repo root
		WithWorkdir("/app/cookie_eaters")            //now we change directory to cookie_eaters containing our .py files

	steps := []string{
		"python -m code.data.B_setup_data",
		"python -m code.data.C_preprocessing",
		"python -m code.features.D_feature_engineering",
		"python -m code.models.F_prepare_train_data",
		"python -m code.models.H_sklearn_train_and_evaluate",
		"python -m code.models.I_save_artifacts",
		"python -m code.models.J_mlflow_model_selection",
		"python -m code.models.K_check_production_model",
		"python -m code.models.L_compare_and_register_model",
	}

	for _, step := range steps {
		log.Println("Running:", step)
		python = python.WithExec([]string{
			"sh", "-c",
			step + " || (echo 'FAILED STEP: " + step + "' && exit 1)",
		})
	}

	_, err = python.ExitCode(ctx)
	if err != nil {
		panic(err)
	}

	_, err = python.
		Directory("artifacts").
		Export(ctx, "../cookie_eaters/artifacts")
	if err != nil {
		panic(err)
	}
}
