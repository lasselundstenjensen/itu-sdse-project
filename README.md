# ITU BDS MLOPS'26 - Project (V2)

## v2 — lecturer tooling (work in progress)

The `v2` branch adds a grading harness: an event engine probes each team's deployed inference endpoint every few minutes for **reachability** and **correctness**, and renders the results as a Claude-Status-style leaderboard. Teams register and edit their endpoint from the dashboard itself using a token issued by the lecturer.

**→ [`docs/v2-dashboard.md`](./docs/v2-dashboard.md)** — dashboard walkthrough (row anatomy, colour semantics, register &amp; edit flow, API surface, mock tokens).

Also worth reading: [`platform/README.md`](./platform/README.md) for how to run the mock locally, [`CLAUDE.md`](./CLAUDE.md) for the full v2 plan, and [`CLARIFY.md`](./CLARIFY.md) for the open questions.

## Quick Start

The project uses a **two-step workflow** for training and deployment:

### Step 1: Training & Registration
```bash
python -m src.pipeline
```

### Step 2: Deployment & Serving
```bash
python -m src.serve_model
```

**See the complete documentation:** [DEPLOYMENT.md](./DEPLOYMENT.md)

## Docker Deployment

The MLflow tracking server, the model serving endpoint, and the training pipeline each run in their own Docker container, so the same images run locally and on DigitalOcean. Inside the containers the tracking server listens on `5000` and the model server on `5001`; only the host port mapping differs between environments.

For local development and quick iteration, run the pipeline directly with `uv` (no Docker needed) — see [Local development](#local-development) below.

### Local development

The fastest loop for development and testing is to run the pipeline natively with `uv`. The pipeline fetches the dataset from the Hetzner object store on first run (no manual data download or DVC pull required), so you only need the tracking server running.

```bash
# Install dependencies
uv sync

# Start the MLflow tracking server (one terminal)
./scripts/start_mlflow_local.sh

# Run the full pipeline (another terminal)
uv run python -m src.pipeline
```

The tracking server defaults to `http://127.0.0.1:5000` (see `src/config.py`). On macOS, port 5000 is used by AirPlay Receiver — disable it in *System Settings → General → AirDrop & Handoff → AirPlay Receiver*, or run the tracking server in Docker where it is mapped to host `5002` (see below).

### Build the images

```bash
docker build -t mlflow-tracking -f docker/Dockerfile.tracking .
docker build -t mlflow-model-serve -f docker/Dockerfile.model-serve .
docker build -t mlflow-train -f docker/Dockerfile.train .
```

### Run the serving stack with Docker Compose (recommended)

This starts the tracking server and model server on a shared bridge network, so the model server reaches the tracking server via Docker's internal DNS (`http://mlflow-tracking:5000`). The `train` service is excluded — it runs on demand (see [Train the model](#train-the-model)).

```bash
docker compose -f docker/docker-compose.yml up -d
```

- Tracking UI: `http://localhost:5002`
- Model server: `http://localhost:5001`

### Run the containers separately

If you prefer to run the containers individually, use `-p` to map ports and `-e MLFLOW_TRACKING_URI` to point the model server at the tracking server. The override path works for any scenario.

Local (Mac, tracking mapped to `5002` to avoid the Spotify conflict):

```bash
docker run -d -p 5002:5000 -v mlflow-data:/mlflow --name mlflow-tracking mlflow-tracking
docker run -d -p 5001:5001 \
    -e MLFLOW_TRACKING_URI=http://host.docker.internal:5002 \
    --name mlflow-model-serve mlflow-model-serve
```

DigitalOcean (tracking mapped to `5000` on the host):

```bash
docker run -d -p 5000:5000 -v mlflow-data:/mlflow --name mlflow-tracking mlflow-tracking
docker run -d -p 5001:5001 \
    -e MLFLOW_TRACKING_URI=http://localhost:5000 \
    --name mlflow-model-serve mlflow-model-serve
```

### Verify

```bash
curl http://localhost:5002          # tracking UI (local)
curl http://localhost:5001/health   # model server
```

#### Inference via `/invocations`

The serving container has no `data/images/raw/` on disk, so send images as
**base64-encoded bytes** (this is the portable contract the model server
expects). The `instances` list is parsed by MLflow and passed to `predict`.

```bash
# base64-encode an image and POST it (matches scripts/test_inference.py)
B64=$(base64 -w 0 data/images/raw/Clean_001.jpg)
curl -s http://localhost:5001/invocations \
    -H "Content-Type: application/json" \
    -d "{\"instances\": [\"$B64\"]}"
# -> {"predictions": [0], "confidences": [0.12], ...}
```

A path string also works **only when the file exists inside the container** —
for local-dev convenience you can `docker cp` an image in and POST its path,
but for any remote/host scenario use base64.

### Port summary

| Environment  | Tracking (host:container) | Model server | Container-to-container              |
|--------------|---------------------------|--------------|-------------------------------------|
| Local (Mac)  | `5002:5000`               | `5001:5001`  | `http://mlflow-tracking:5000`       |
| DigitalOcean | `5000:5000`               | `5001:5001`  | `http://mlflow-tracking:5000`       |

### Train the model

The training container runs the full pipeline (fetch data → train → register → deploy) and logs everything to the tracking server. It is gated behind the `train` Compose profile, so it does not start with the serving stack — run it on demand.

```bash
# Train against the running serving stack (Compose handles networking + MLflow URI)
docker compose -f docker/docker-compose.yml run --rm train
```

The training image fetches the dataset from the Hetzner object store itself on first run, so no data needs to be mounted or pre-staged. It uses the same `python:3.13-slim` base as the model server, so a model trained in this container serves without a Python-version mismatch.

To run training standalone (without Compose), point it at a tracking server with `-e MLFLOW_TRACKING_URI`:

```bash
docker run --rm -e MLFLOW_TRACKING_URI=http://host.docker.internal:5002 mlflow-train
```

## Task

Based on the input provided (see below), fork the repository and restructure the code to adhere to the concepts and ideas you have seen throughout the course.  The diagram below provides a detailed overview of the structure that the solution is expected to follow.   

![Project architecture](./docs/project-architecture.png)

For the exam submission, we expect you to submit a pdf containing:
- the list of members of the group
- the link to the github.com public repository hosting your solution
  - following the above, there is *no need* to invite the teaching staff as collaborators

The repository linked in the submission should contain:

- A README.md file that describes the project
- GitHub automation workflow
- Dagger workflow (in Go)
- All history


## Inputs

You are given the following material:
- Python monolith (see `notebooks` folder)
- Raw input data (see `notebooks/artifacts` folder)
- GitHub action to test model inference (see [`model-validator`](https://github.com/lasselundstenjensen/itu-sdse-project-model-validator) action)

## Outputs

- Your GitHub repository (including all history)
  - A README.md file that describes the project
  - GitHub automation workflow
  - Dagger workflow (in Go)
- Model artifact produced by GitHub workflow and named 'model'

> **NOTE:**
> The Dagger workflow can be run locally or inside the GitHub workflow—both are viable options during development.
>
> The Dagger workflow can run locally and can also be made to produce outputs locally during development. But when wrapping the Dagger workflow in a GitHub workflow, the output is instead stored inside the GitHub runner (i.e. a virtual machine).
>
> Use the publicly available [`actions/upload-artifact`](https://github.com/actions/upload-artifact) to store the model artifact in the GitHub worklow pipeline.
>
> This model artifact can then be picked up by the [action provided](https://github.com/lasselundstenjensen/itu-sdse-project-model-validator), which will run some inference tests to ensure that the correct model was trained.


## How will we assess

Below, we provide information on how we will assess the submission clustered around several aspects.  The list relates to groups of size 3; if your group is of size 4, you are expected also to work on the optional items, i.e., to use pull requests and to provide tests.

#### Versioning

- Use of Git (semantic commit messages, branches, branch longevity, commit frequency/size)
- Management of data
- Use of pull requests (OPTIONAL)

#### Programming

- Decomposition of Python notebook
- Adherance to standard data science MLOps project structure
- Presence of tests (OPTIONAL)

#### Workflow automation

- Presence of a workflow that trains the model
- Presence of a workflow that tests the model
- Structure of Dagger workflow
- Orchestration of Dagger workflow through GitHub workflow

#### Documentation (README.md)

- Description of project structure
- How to run the code and generate the model artifact


## Questions

If you have any questions about the information shared here, please feel free to post them on Learnit. Answers to private emails on this topic will also be shared on Learnit, along with the original email content, so that everyone has access to the same information.
