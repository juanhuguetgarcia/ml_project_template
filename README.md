basic_ml_project
==============================

A basic template for a ML project based in https://github.com/drivendata/cookiecutter-data-science

ML Productization
-----------------
Boston house prediction.

The problem has been framed as a regression problem.

Final solution consists of a data transformation pipeline where all the data processing is handled from one single function call.

This allows to keep the code more legible and favors maintenance as dependencies between functions are well stated in a single file.

The solution consists of three different scripts:

1. A dataset creation script that generates the boston houses data from sklearn basic datasets and splits in train and test. Test file does not have target column.
1. A training script that automatizes a simple training and evaluation process over the modelling data. 
1. A predict script that allows to load one pretrained model and use it to give predictions over new data. Both, model and data file to run the predictions can be changed.
1. A simple APIRest server that allows to give predictions in real time. The API has been build using the Flask framework and can be easily scaled to serve large requests volumes.

Also, an example of data to send to the APIRest is given in json format.


Quick start
===========

Dependencies
------------
To be able to run the scripts, install the dependencies specified in the `environment.yml` file. This will create
a virtual environment under the name `basic_ml_project`

```bash
conda env create -f environment.yml
```

Make dataset
------------
A util to create a simple dataset from boston houses into modelling and test subsets

`bash make data`

It calls

`python src/data/make_dataset.py`


Train model
-----------

A util to train a simple model on modelling subset.

```bash
make train_model
```

It calls

`python src/models/train_model.py`


Batch prediction
----------------

A util to run batch inference on a given dataset and a given model

```bash
make batch_predict
```

It calls

`python src/models/predict_model.py`

API
---
A util to set up a simple API-REST using flask is provided

```bash
make launch_api
```

Test API
--------
A util to send a `POST` request to the api is also provided. It request's the endpoint with the json example
in `example_query_api.json`


```bash
make request_predict
```

Project Organization
--------------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── api           <- Simple flask api to serve models via api-rest
    │   │   └── app.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


TODO
====
- [ ] Add cli to train, predict and api to specify via the command line the dependencies
- [ ] Add Dockerfile to allow reproducibility
- [ ] Add testing

--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
