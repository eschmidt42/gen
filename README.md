# `gen` 

`gen` contains some (variational) auto encoder models.

## Why?!1?

Really just for tinkering with generative ML models.

## Setup

This environment for this package is managed using poetry, see [this page](https://python-poetry.org/docs/cli) for an overview over the commands.

To get started first clone the github repo and then cd into it. In the following execute:

`poetry install`: sets up the environment

`poetry run pre-commit install`: sets up pre-commit (does a few basic checks after each commit, like verifying empty notebooks and black formatting, see `.pre-commit-config.yaml`)

`poetry run pre-commit run --all-files`: runs all checks once over all files

`poetry run jupyter notebook --no-browser --port 8888`: starts a jupyter notebook server

## How to use

Via _jupyter_:
* `vae.ipynb` to play with VAE
* `ae.ipynb` to play with AE

Via _streamlit_:
* run `poetry run streamlit run app.py` and open `localhost:8501` in your browser to interact with both models and toy with configs

## TODOS

* add demonstrations where AEs don't work (slightly off input sample space) to highlight the need for VAEs
* test usefulness of generative models to create useful features, e.g. for tabular kaggle data
* add sensible tests for generative models that can be used for `unittest`
