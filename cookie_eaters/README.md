# Cookie Eaters - MLOps and Software Engineering Project

## Project overview
This is our MLOps project for the 'Data Science in Production: MLOps and Software Engineering' course at the IT-University of Copenhagen. We have implemented a ML pipeline that runs inside containers using Dagger to ensure reproducibility across machines.

## Reproducibility

Why This Is Reproducible:
- Dependencies are specified in `requirements.txt`
- The pipeline is fully defined in `dagger/pipeline.go`
- Execution environment is containerized (with Docker)

## How to run the workflow

### Option 1: GitHub Actions(CI workflow)
GitHub Actions automatically validates reproducibility on every push and pull request to `main`. This ensures the pipeline executes successfully in a clean environment and produces deterministic outputs.Following is the Github Actions Continuous Integration (CI) workflow:

1. Checks out the repository
2. Sets up Go (version from dagger/go.mod)
3. Runs Dagger inside Docker
4. Executes the full pipeline
5. Exports the trained model artifact
6. Runs an external model validation action

Go to Github repo and navigate to Actions tab, select the workflow and main branch, then tab run workflow.

### Option 2: Locally 
In order to run locally, you want to make sure to have the following installed:

1. Python (3.13.3)
2. Go (1.24.0) - specified in `dagger/go.mod`
3. Dagger CLI (v0.18.16)
4. Docker Engine - required for the pipeline to run in containers

Note: you must have the Docker Engine open and runnning to execute the Dagger pipeline locally

Running these commands on any machine will reproduce the full training pipeline and generate identical artifacts:

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
The pipeline reads `raw_data.csv`, performs preprocessing, trains models, tracks results with MLflow and saves artifacts.

## Project organization
The following structure highlights the main architectural components of the repository initiated with CCDS and redesigned to our project needs:

```
itu-sdse-project/
├── .dvc/                         <- data versioning
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
    ├── pipeline.go               <- Containerized pipeline definition
    ├── go.mod                    <- Go dependencies 
    └── go.sum                    <- Go dependency integrity (verifies dependencies installed)

```