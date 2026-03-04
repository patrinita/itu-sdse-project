# Cookie Eaters - MLOps and Software Engineering Project

Amina Lykke Said & Patricia Nita

Repo: https://github.com/patrinita/itu-sdse-project

## Project overview
This project implements an end-to-end, reproducible ML pipeline that:
- prepares raw data
- trains and evaluates ML models
- tracks experiments with MLflow
- produces a model artifact (`model.pkl`) via CI

#### The pipeline runs inside containers using Dagger to ensure reproducibility across machines.

## Quick start
```bash
git clone https://github.com/patrinita/itu-sdse-project.git
cd itu-sdse-project/dagger
go mod download
dagger run go run pipeline.go
```
Generated artifacts will appear in:
```bash
cookie_eaters/artifacts/
```

## Reproducibility
This project is fully reproducible. The entire ML pipeline runs inside a Docker container orchestrated by Dagger. No local Python setup is required. 
The raw dataset is versioned in Git at `cookie_eaters/raw/raw_data.csv`. DVC is initialized in the repository but not used by the current pipeline.

Prerequisites
- Docker Desktop (running)
- Go (version specified in `dagger/go.mod`)
- Dagger CLI

Why this is reproducible:
- Execution environment is containerized (Docker)
- Python version is fixed via `python:3.11-slim`
- Dependencies are pinned in `requirements.txt`
- The pipeline is fully defined as code in `dagger/pipeline.go`
- No manual setup or local Python installation is required.

#### Running the Quick Start commands on any machine with Docker installed will reproduce the full training pipeline and generate the model artifact.

## Continuous Integration (CI)
GitHub Actions automatically validates reproducibility on every push and pull request to `main`.
The CI workflow:
1. Checks out the repository
2. Sets up Go (version from dagger/go.mod)
3. Runs Dagger inside Docker
4. Executes the full pipeline
5. Exports the trained model artifact
6. Runs an external model validation action

### This ensures the pipeline executes successfully in a clean environment and produces the expected model artifact for validation.

## Pipeline overview
#### The pipeline reads `raw_data.csv`, performs preprocessing, trains models, tracks results with MLflow and saves artifacts.

## Project organization
The following structure highlights the main architectural components of the repository and how responsibilities are separated across data versioning, ML logic, orchestration, and CI:

```
itu-sdse-project/
├── .dvc/                         <- DVC initialized (not used for current dataset)
├── .github/workflows/            <- CI workflow (GitHub Actions)
├── cookie_eaters/                <- Python ML project
│   ├── code/                     <- Pipeline step implementations (data, features, models)
│   ├── cookie_eaters/            <- Python package (importable module)
│   ├── raw/
│   │   └── raw_data.csv
│   ├── artifacts/                <- Generated outputs (models, metrics)
│   ├── mlruns/                   <- MLflow experiment tracking
│   └── requirements.txt
├── dagger/                       <- Dagger orchestration layer
│   ├── pipeline.go               <- Containerized pipeline definition
│   ├── go.mod
│   └── go.sum
├── docs/                         <- Architecture diagrams
├── notebooks/                    <- Exploration & inference
├── action.yml                    <- Custom GitHub Action
└── README.md
```