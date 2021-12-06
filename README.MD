# Health Outcome Pathway Prediction

This work was developed under the umbrella of the project [FrailCare.AI](https://frailcareai.vohcolab.org/) within [VOH.CoLAB](https://vohcolab.org/) for the conclusion of a masters degree.

## Dissertation abstract 
This dissertation is part of the project FrailCare.AI, which aims to detect frailty in the elderly Portuguese population in order to optimize the SNS24 (telemonitoring) service, with the goal of suggesting health pathways to reduce the patients frailty. Frailty can be defined as the condition of being weak and delicate which normally increases with age and is the consequence of several health and non-health related factors. 

A patient health journey is recorded in EHR, which are rich but sparse, noisy and multi-modal sources of truth. These can be used to train predictive models to predict future health states, where frailty is just one of them. In this work, due to lack of data access we pivoted our focus to phenotype prediction, that is, predicting diagnosis. What is more, we tackle the problem of data-insufficiency and class imbalance (e.g. rare diseases and other infrequent occurrences in the training data) by integrating standardized healthcare ontologies within graph neural networks. We study the broad task of phenotype prediction, multi-task scenarios and as well few-shot scenarios - which is when a class rarely occurs in the training set. Furthermore, during the development of this work we detect some reproducibility issues in related literature which we detail, and also open-source all of our implementations introduding a framework to aid the development of similar systems.


## Deliverables
1. New baselines for different label schemes (different level of granularity), for both MIMIC 3 and eICU
2. Easy-to-use script to reproduce all results
3. Extensible and modular architecture
4. Tools to handle the studied datasets




### Project Structure

#### Notebooks
All .ipynb used for experiments and exploratory analysis

Notebooks that end with - Gradio, are interactive.
* Results - Gradio.ipynb offers a simple way to explore the results of all experiments
* What if - Gradio.ipynb uses a trained model to explore predictions


#### dataset_parsers
Dataset parsers for MIMIC III and eICU, based on the interface defined in base.py.

#### graph_builders
Classes responsable for creating the graphs given a set of EHRs of one or multiple patients.

#### Datasets
Place where the datasets should be placed in. Here, there is only a really small randomized subset of the MIMIC 3 Dataset for demonstration online.

#### Others
models.py - PyTorch model implementation

phenotype.py - Script used to run all experiments

Processing.py, metrics.py and utilities.py hold auxiliary functions.

ICDCodesGrouper.py is responsible for mapping ICD9 codes to different levels of granularity - CCS, Category Level, 3-digit and ICD9 Chapters.
