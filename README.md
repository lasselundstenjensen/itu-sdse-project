# ITU BDS MLOps'25 - Lead Prediction Project

## Group Members

- Dmytro Stavnychyi
- Vitus Andreas Moustgaard
- Ada Yıldız

## Project Description

This project implements a complete MLOps pipeline for predicting lead conversion indicators using machine learning. The pipeline includes:

- **Data versioning** with DVC
- **Model training** with XGBoost and Logistic Regression
- **Experiment tracking** with MLflow
- **Container orchestration** with Dagger (Go)
- **CI/CD automation** with GitHub Actions

The best performing model (Logistic Regression, F1~0.78) is automatically trained, validated, and deployed as a GitHub artifact.

## Project Structure

```
.
├── .github/workflows/       # GitHub Actions CI/CD
│   └── train-model.yml     # Model training workflow
├── mlops_project/          # Main Python package
│   ├── data/              # Data loading and preprocessing
│   │   ├── load.py        # DVC data pulling and loading
│   │   └── preprocess.py  # Data cleaning and preprocessing
│   ├── deployment/
│   │   ├── mlflow_utils.py  # Util functions for mlflow
│   ├── features/          # Feature engineering
│   │   └── build_features.py
│   ├── models/            # Model training and evaluation
│   │   ├── train.py       # XGBoost & Logistic Regression training
│   │   └── evaluate.py    # Model evaluation and comparison
│   └── scripts/
│       └── main.py        # Main pipeline orchestration
│   └── utils/
│       └── helpers.py     # Helpers for numeric columns and missing values
│   └── config.py
├── notebooks/             # Original exploratory notebooks
│   └── artifacts/         # Raw data (DVC tracked)
├── output/                # Generated artifacts (gitignored)
│   ├── model              # Best model (for GitHub workflow)
│   ├── results.json       # Evaluation metrics
│   └── *.pkl              # Serialized models
├── mlruns/                # MLflow experiment tracking
├── dagger-pipeline.go     # Dagger orchestration (Go)
├── requirements.txt       # Python dependencies
├── .dvc/                  # DVC configuration
└── README.md
```

**Note:** The project follows the [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) template structure. Directories not listed above (`models/`, `tests/`, `reports/`, `references/`, `docs/`) are part of the original template but were not actively used in this implementation, as they were not required for the core MLOps pipeline.

## Prerequisites

- **Python 3.10**
- **Go 1.21+**
- **Dagger CLI**
- **DVC**
- **Git**

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Vitus-A-M/MLOps-project.git
cd MLOps-project
```

### 2. Set up Python environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install DVC

```bash
pip install dvc
```

### 4. Pull data

```bash
dvc pull
```

## How to Run

### Option 1: Direct Python Execution

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the pipeline
python mlops_project/scripts/main.py
```

The model artifacts will be saved in `output/`:

- `output/model` - Best model (no extension)
- `output/xgboost_model.pkl` - XGBoost model
- `output/logistic_regression_model.pkl` - Logistic Regression model
- `output/results.json` - Evaluation metrics
- `output/best_model.txt` - Model metadata

### Option 2: Using Dagger (Containerized)

```bash
# Install Dagger CLI
curl -L https://dl.dagger.io/dagger/install.sh | sh

# Run pipeline in container
dagger run go run dagger-pipeline.go
```

### Option 3: Automated with GitHub Actions

Push to `main` branch or any `feature/*` branch:

```bash
git add .
git commit -m "feat: your changes"
git push origin main
```

GitHub Actions will automatically:

1. Pull data with DVC
2. Run Dagger pipeline
3. Train models
4. Upload `model` artifact

## MLOps Components

### Data Versioning (DVC)

- **Raw data**: `notebooks/artifacts/raw_data.csv`
- **Remote**: GitHub repository
- **Commands**:
  - Pull data: `dvc pull`
  - Check status: `dvc status`

### Experiment Tracking (MLflow)

- **Location**: `mlruns/` directory
- **Features**: Automatic parameter and metric logging
- **View experiments**:

```bash
  mlflow ui
  # Open http://localhost:5000
```

### Orchestration (Dagger)

- **File**: `dagger-pipeline.go`
- **Container**: Python 3.10
- **Benefits**:
  - Reproducible builds
  - Cross-platform compatibility
  - Isolated environments

### CI/CD (GitHub Actions)

- **Workflow**: `.github/workflows/train-model.yml`
- **Triggers**: Push to `main` or `feature/*` branches
- **Steps**:
  1. Checkout code
  2. Set up Python 3.10
  3. Install DVC and pull data
  4. Set up Go and Dagger
  5. Run Dagger pipeline
  6. Upload model artifact

## Model Performance

Current best model: **Logistic Regression**

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.79  |
| F1 Score  | 0.78  |
| Precision | 0.79  |
| Recall    | 0.78  |

Detailed metrics available in `output/results.json` after training.

## Development Workflow

### Branch Strategy

- `main` - Production-ready code
- `feature/*` - Feature development -`debug/*` - Debugging existing features

### Semantic Commits

We use descriptive commit messages that clearly explain changes:

**Examples from our project:**

- "add GitHub Actions workflow for model training"
- "fix DVC data pulling in container"
- "update main.py with comprehensive error handling"

While we don't strictly follow conventional commit prefixes (`feat:`, `fix:`), our messages clearly describe what was changed and why.

### Making Changes

1. Create feature branch:

```bash
   git checkout -b feature/your-feature
```

2. Make changes and test locally:

```bash
   python mlops_project/scripts/main.py
```

3. Commit with semantic message:

```bash
   git add .
   git commit -m "feat: add your feature"
```

4. Push and create PR:

```bash
   git push origin feature/your-feature
```

5. GitHub Actions will run automatically

## Troubleshooting

### DVC pull fails

```bash
# Fallback: manual download
curl -L https://raw.githubusercontent.com/Jeppe-T-K/itu-sdse-project-data/refs/heads/main/raw_data.csv -o notebooks/artifacts/raw_data.csv
```

### Dagger fails locally

```bash
# Check Docker is running
docker ps

# Clean Dagger cache
dagger run --cleanup
```

### Import errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Repository

**GitHub**: https://github.com/Vitus-A-M/MLOps-project

## License

This project is part of the ITU BDS MLOps course (2025).

---

**Questions?** Contact the team members or post on the course forum.
EOF
