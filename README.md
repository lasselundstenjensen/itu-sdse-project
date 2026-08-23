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
python -m src.deployment.deploy_and_serve
```

**See the complete documentation:** [DEPLOYMENT.md](./DEPLOYMENT.md)

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
