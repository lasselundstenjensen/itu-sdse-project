# ITU BDS MLOPS'26 - Project (V2)

## Task

Based on the input provided (see below), fork the repository and restructure the code to adhere to the concepts and ideas you have seen throughout the course. The diagram below provides a detailed overview of the structure that the solution is expected to follow.

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
- Python monoliths (see `notebooks` folder)
- Raw input data (pulled from the object store by the training notebook)
- Group login to https://sdse-event-engine-r5ih8.ondigitalocean.app/ for monitoring deployed model
- DigitalOcean App Platform spec for the deployment (see `infra/digitalocean/app.yaml`)

## Outputs

- Your GitHub repository (including all history)
  - A README.md file that describes the project
  - GitHub automation workflow
  - Dagger workflow (in Go)
- Model artifacts produced by GitHub workflow
- Hosted endpoint in DigitalOcean
  - /health that returns {"status": "ok"}
  - /invocations that take binary image payload
  - /drift that returns model/data drift information

> **NOTE:**
> The Dagger workflow can be run locally or inside the GitHub workflow—both are viable options during development.
>
> The Dagger workflow can run locally and can also be made to produce outputs locally during development. But when wrapping the Dagger workflow in a GitHub workflow, the output is instead stored inside the GitHub runner (i.e. a virtual machine).
>
> Use the publicly available [`actions/upload-artifact`](https://github.com/actions/upload-artifact) to store the model artifact in the GitHub worklow pipeline.
>
> This model artifact can then be picked up by the [action provided](https://github.com/lasselundstenjensen/itu-sdse-project-model-validator), which will run some inference tests to ensure that the correct model was trained.

## How will we assess

Below, we provide information on how we will assess the submission clustered around several aspects. The groups are expected to be of size 4, and you're expected to use pull requests and tests.

### Versioning

- Use of Git (semantic commit messages, branches, branch longevity, commit frequency/size)
- Management of data and artifacts
- Use of pull requests

### Programming

- Decomposition of Python notebook
- Adherance to standard data science MLOps project structure
- Presence of tests

### Workflow automation

- Presence of a workflow that trains the model
- Presence of a workflow that tests the model
- Structure of Dagger workflow
- Orchestration of Dagger workflow through GitHub workflow

### Documentation (README.md)

- Description of project structure
- How to run the code and generate the model artifact

## Questions

If you have any questions about the information shared here, please feel free to post them on Learnit. Answers to private emails on this topic will also be shared on Learnit, along with the original email content, so that everyone has access to the same information.

# cc_project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

cookiecutter setup for this thing

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         cc_project and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── cc_project   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes cc_project a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
