# Cookie Eaters - MLOPs and Software Engineering Project

# Amina Lykke Said & Patricia Nita

**Repo: https://github.com/patrinita/itu-sdse-project**

## Project overview
This project implements an end-to-end ML pipeline with the goal to build a reproducible automated system that: takes raw_data as input, preprocesses and prepares it, trains and evaluates ML models, tracks experiments and metrics using MLflow and stores model artifacts such that reproducibility is ensured.

We use Dagger to run the pipeline inside a container, which helps ensure reproducibility across different machines.

## How to run the project
Prerequisites:
- Docker (required to run the pipeline with Dagger)
- Go
- All Python dependencies including MLflow will be installed automatically inside the container when running the pipeline

## How to run the pipeline
Run these commands in the repository root:
```bash
cd dagger
go run pipeline.go
```
This will run the full ML pipeline inside a container and will generate its outputs in the `cookie_eaters/artifacts/` and `cookie_eaters/mlruns/` directories.

## Data setup: do this before running the code!
The raw dataset is required to run the pipeline and was not committed to Git.

The data file must be placed locally at: cookie_eaters/raw/raw_data.csv

This folder is gitignored and must be created locally.

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Local execution
Some pipeline steps can also be run locally when debugging the code.  
These steps were chosen because they represent the main stages of the pipeline (data setup, preprocessing, feature engineering, training data preparation, model training) and they can be tested independently:
```bash
cd cookie_eaters
pip install -r requirements.txt
python -m code.data.B_setup_data
python -m code.data.C_preprocessing
python -m code.features.D_feature_engineering
python -m code.models.F_prepare_train_data
python -m code.models.H_sklearn_train_and_evaluate
```
## Pipeline overview
The pipeline starts from a raw CSV file (`cookie_eaters/raw/raw_data.csv`) and performs data preparation and preprocessing steps, followed by feature engineering and model training.  
During training, the model performance is evaluated and tracked using MLflow.  
The final outputs (including processed datasets, trained models and evaluation results) are saved as artifacts (inside `cookie_eaters/artifacts/`).

## Comments on reproducibility
The pipeline is executed inside a container using Dagger to ensure consistent execution across environments. Generated outputs such as the artifacts and the MLflow runs are excluded from version control to keep the repository clean and reproducible.

## Project organization
To make it easier to see where things are located:
```
├── .github/workflows/        <- GitHub Actions workflows
├── cookie_eaters/
│   ├── raw/                 <- Raw input data (gitignored)
│   ├── code/                <- Main pipeline implementation
│   │   ├── data/            <- Data loading & preprocessing
│   │   ├── features/        <- Feature engineering
│   │   └── models/          <- Training, evaluation, model registry logic
│   ├── artifacts/           <- Generated outputs (gitignored)
│   ├── mlruns/              <- MLflow tracking (gitignored)
│   ├── requirements.txt
│   └── __init__.py
├── dagger/
│   ├── pipeline.go          <- Dagger pipeline definition
│   ├── go.mod
│   └── go.sum
├── notebooks/               <- Exploratory notebooks
├── README.md
└── .gitignore
```