package main //defines an executable program

import (
	"context"       //to control execution tasks
	"log"           //to log messages
	"path/filepath" //to navigate in filesystem

	"dagger.io/dagger" //to create container, pipeline and workflow innside Go code
)

func main() { //initiales the program (entry point)
	ctx := context.Background() // creates the executable enviroment (base)

	client, err := dagger.Connect(ctx, dagger.WithLogOutput(log.Writer())) //connetcs te program to Dagger engine and enable logging to terminal
	if err != nil {
		panic(err) // if error occurs then stop the execution and raise the error
	}
	defer client.Close() //when the pipelinne finishes then close the conection to Dagger Engine

	// Get absolute path to project root
	absPath, err := filepath.Abs("../") //from dagger folder we directs to the project root (converts relative path to absolute path)
	if err != nil {
		panic(err)
	}

	// Mount project root
	src := client.Host().Directory(absPath) //take the directory from our host computer and make it available to the container

	python := client.Container().
		From("python:3.11-slim").          //create a container using python:3.11-slim
		WithMountedDirectory("/app", src). //this is where the container stores the directory

		//we install all necessary libaries listed in the requirements.txt
		WithWorkdir("/app/cookie_eaters").
		WithExec([]string{"pip", "install", "--default-timeout=1000", "--no-cache-dir", "-r", "requirements.txt"})

		// defines the steps as a list of commands for the pipeline to run
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
	// loops through every element in the steps list one by one - intentionally ignores the index (0, 1, 2...) because we never use the index
	for _, step := range steps {
		log.Println("Running:", step) //logs the step aka print timestamps + messages to the terminal
		//run the command in container
		python = python.WithExec([]string{
			"sh", "-c", //tell it to run inside shell
			step + " || (echo 'FAILED STEP: " + step + "' && exit 1)", //if step fails then it print error + stop pipeline and return error code
		})
	}

	//Runs the full container pipeline
	_, err = python.ExitCode(ctx)
	if err != nil {
		panic(err)
	}

	_, err = python.
		Directory("artifacts").
		Export(ctx, absPath+"/cookie_eaters/artifacts")
	if err != nil {
		panic(err)
	}
}
