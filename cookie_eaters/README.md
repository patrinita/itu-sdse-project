# Cookie Eaters - MLOps and Software Engineering Project

Amina Lykke Said & Patricia Nita

Repo: https://github.com/patrinita/itu-sdse-project

## Project overview
This project implements an end-to-end, reproducible ML pipeline that:
- prepares raw data
- trains and evaluates ML models
- tracks experiments with MLflow
- produces a versioned model artifact

The pipeline runs inside containers using Dagger to ensure reproducibility across machines.

## Requirements
- Docker
- Dagger
- Go (version defined in `dagger/go.mod`)

## Run the pipeline
From the repo root:

```bash
dagger run go run ./dagger
```

This runs the full pipeline (test -> train -> build) and outputs the `model` artifact.

## Pipeline overview
The pipeline reads `cookie_eaters/raw/raw_data.csv`, performs preprocessing and feature engineering, trains models, tracks results with MLflow and saves outputs in `cookie_eaters/artifacts/`.

## Reproducibility
All steps run inside containers via Dagger. Generated artifacts and MLflow runs are excluded from version control.

## Project organization
To make it easier to see where things are located:

```
itu-sdse-project/
├── .github/workflows/
├── cookie_eaters/
│   ├── code/                       <- Pipeline steps
│   │   ├── data/                   <- Data setup & preprocessing
│   │   ├── features/               <- Feature engineering
│   │   └── models/                 <- Training, evaluation, MLflow & model registry
│   ├── cookie_eaters/
│   │   └── __init__.py
│   ├── raw/
│   ├── .gitignore
│   ├── Makefile
│   ├── pyproject.toml
│   ├── README.md
│   ├── requirements.txt
│   └── setup.cfg
├── dagger/
│   ├── pipeline.go
│   ├── go.mod
│   └── go.sum
├── docs/
├── notebooks/
├── .gitignore
├── action.yml
└── README.md
```
