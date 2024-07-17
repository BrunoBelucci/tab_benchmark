# tab_benchmark

**tab_benchmark** is a Python package for benchmarking tabular data classification and regression models. It is 
designed to be easy to use and to work with a wide variety of classical and deep learning models. It is also designed 
to be easy to extend with new models and datasets.  Besides, it also includes a hyperparameter optimization module.

The advantage of using **tab_benchmark** is that all the models have the same API, so you only need to learn how 
to use one model to use all of them. 

We have also included a module for downloading datasets from various sources, including Kaggle, but we mainly use
openml datasets.

**tab_benchmark** is built following the [scikit-learn](https://scikit-learn.org/stable/) API. 

This is the second version of the package, hopefully this time it will be more user-friendly and easier to extend and
maintain.

## Installation

You can install tab_benchmark using pip or conda. We recommend using conda because it seems to be more reliable in
solving the dependencies. 
### conda
We recommend using the mamba solver because it is faster than the default conda solver. To configure mamba as the
default solver, run:
```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```
For more information about mamba, see [here](https://conda.github.io/conda-libmamba-solver/).

To install tabular_benchmark using conda in a new environment, run:
```bash
conda env create -f environment.yml
```

### pip
You can install tab_benchmark with pip (if pip has problems solving the dependencies try using 
python 3.10.*, for example using a conda environment)

```bash
pip install tab_benchmark
```

or you can install it in development mode:

```bash
pip install -e .
```

to run the tests
```bash
pip install -e .[test]
```

### Dependencies

- pytorch (torch via pip)
- lightning
- matplotlib
- numpy
- pandas
- scikit-learn
- scikit-posthocs
- scipy
- py-xgboost (xgboost via pip)
- lightgbm
- catboost
- dask
- einops
- mlflow
- postgresql
- pytest
- ipykernel
- ipywidgets
- sphinx
- sphinx_rtd_theme
- openml

## Usage

Take a look at notebooks in the examples folder.

## Supported models

- [xgboost](https://xgboost.readthedocs.io/en/stable/index.html)
- [lightgbm](https://lightgbm.readthedocs.io/en/stable/)
- [catboost](https://catboost.ai/en/docs/)
- [tabnet](https://github.com/dreamquark-ai/tabnet)
- [tabtransformer](https://github.com/lucidrains/tab-transformer-pytorch)
- [saint](https://github.com/somepago/saint)
- [node](https://github.com/Qwicen/node)

## Supported datasets

TO DO


## Supported hyperparameter optimization algorithms
TO DO

## License

[MIT](https://choosealicense.com/licenses/mit/)


TO DO UPDATE THIS SECTION
## Code Development Description

- models in directory models:
  - Final model that the user interact with, should follow scikit-learn API, define reasonable defaults

- dnn models:
  - The architecture of the dnn must be included in the folder DNNs/architectures and it also must include the 
    static method tabular_dataset_to_model_kwargs that initialize the model from a tabular dataset (defining data 
    related parameters such as input size, output size, etc.).
  - The train loop should be defined by the base_step method in the lightning module that this models use (in 
    DNNs/modules). The default optimizers, schedulers and callbacks must be defined in the model file.

- Code Style/docstrings:
  - We try to always use the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
  - This link is a good reference for docstrings: [Google docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
  - We also use type annotations whenever possible (https://docs.python.org/3/library/typing.html and https://peps.python.org/pep-0484/)

TO DO UPDATE THIS SECTION
## How to add a new model
- Start by creating a file with the model name in models.
- Create a class with the name of the model and inherit from the BaseModel class in models/base_model.py. If the 
  model is a DNN or a GBDT you can inherit from DNNModel (models/dnn_base_model) or GBDTModel (models/gbdt_base_model).
- Implement all the abstract methods in the base class. Do not forget to call `super().__init__(**needed_arguments**)`
  in 
  the `__init__` method and `super().on_fit_start(**neded_arguments**)` `super().on_fit_end(**neded_arguments**)` in 
  the 
  `fit` method.
- If the model is a DNN, create a file with the name of the architecture that it uses in DNNs/architectures. Inherit 
  from the BaseArchitecture class in DNNs/architectures/base_architecture.py. Implement all the abstract methods in
  the base class. The forward method should return a dictionary with the outputs of the model and a key called 
  'y_pred'. If needed, implement a custom LightningModule inheriting from TabularModule in 
  DNNs/modules/TabularModule.py and adapt as needed.
- Test the model. Add a test_moodel_name.py to the tests folder. You can follow the approach used in the other files.

TO DO UPDATE THIS SECTION
## REALLY Slow datasets/models (close or more than 1 day with default parameters and tesla P4 gpu, 10 cores Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz)
### XGBoost
- aloi
- dionis

### CatBoost
- dionis
- aloi
- poker

### Node
- covertype
- dionis
- allstate_claim_predictions
- poker

### Saint
- covertype
- dionis
- allstate_claim_predictions
- aloi

### TabTransformer
- allstate_claim_predictions

## Slow datasets/models (more than 1 hour with default parameters and tesla P4 gpu, 10 cores Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz)
### CatBoost
- albert
- allstate_claim_predictions
- connect-4	
- dilbert
- ldpa

### NODE
- adult
- albert
- amazon_employee
- aps_failure
- bank_marketing_old
- connect-4	
- credit_card_fraud
- dilbert
- helena
- higgs_kaggle
- jannis
- ldpa
- miniboone	
- nomao
- numerai
- shuttle
- skin_segmentation
- volkert
- walking
- year_prediction

### Saint
- albert
- aps_failure
- dota2
- higgs_kaggle
- jannis
- kdd2009_appetency
- kdd2009_churn	
- kdd2009_upselling
- miniboone
- poker
- skin_segmentation
- volkert
- year_prediction

### TabNet
- allstate_claim_predictions
- poker

### TabTransformer
- albert
- covertype
- poker

## Out of Memory Error (process killed)
### Saint
- cnae-9 
- christine 
- dilbert 
- fabert
- arcene
- ujiindoorloc_longitude
- ujiindoorloc_latitude
- relative_location_of_ct

## Even smaller model cannot be trained with P4 gpu
### TabTransformer
- arcene

### Saint
- christine

## Every model can run:
iris, wine, breast_cancer, adult, anneal, aps_failure, australian_credit, bank_marketing_old, blood_transfusion, car, cnae-9, connect-4, german_credit, dota2, htru2, insurance_company, internet_usage, kr-vs-kp, ldpa, mfeat-fac, miniboone, mushroom, nomao, online_shoppers, qsar_bio, image_segmentation, seismic-bumps, shuttle, skin_segmentation, spambase, vehicle, walking, albert, helena, jannis, jasmine, philippine, sylvine, volkert, amazon_employee, higgs_kaggle, blastchar, credit_card_fraud, numerai, shrutime, phoneme, kdd2009_appetency, kdd2009_churn, kdd2009_upselling, diabetes, year_prediction, sarcos, california_housing, airfoil_self_noise, yacht_hydrodynamics, concrete_slump_test_slump, concrete_slump_test_flow, concrete_slump_test_compressive_strength, computer_hardware, concrete_compressive_strength, csm, physicochemical_properties_of_protein, online_video_characteristics, kegg_metabolic_directed, communities_and_crime

## Every model can run "fast" (less than 1 hour for all models):
iris, wine, breast_cancer, anneal, australian_credit, blood_transfusion, car, cnae-9, german_credit, htru2, insurance_company, internet_usage, kr-vs-kp, mfeat-fac, mushroom, online_shoppers, qsar_bio, image_segmentation, seismic-bumps, spambase, vehicle, albert, jannis, jasmine, philippine, sylvine, blastchar, shrutime, phoneme, diabetes, sarcos, california_housing, airfoil_self_noise, yacht_hydrodynamics, concrete_slump_test_slump, concrete_slump_test_flow, concrete_slump_test_compressive_strength, computer_hardware, concrete_compressive_strength, csm, physicochemical_properties_of_protein, online_video_characteristics, kegg_metabolic_directed, communities_and_crime

## Every model excluding Saint and Node can run:
covertype, christine, dilbert, relative_location_of_ct, ujiindoorloc_longitude, ujiindoorloc_latitude, iris, wine, breast_cancer, adult, anneal, aps_failure, australian_credit, bank_marketing_old, blood_transfusion, car, cnae-9, connect-4, german_credit, dota2, htru2, insurance_company, internet_usage, kr-vs-kp, ldpa, mfeat-fac, miniboone, mushroom, nomao, online_shoppers, qsar_bio, image_segmentation, seismic-bumps, shuttle, skin_segmentation, spambase, vehicle, walking, albert, helena, jannis, jasmine, philippine, sylvine, volkert, amazon_employee, higgs_kaggle, blastchar, credit_card_fraud, numerai, shrutime, phoneme, kdd2009_appetency, kdd2009_churn, kdd2009_upselling, diabetes, year_prediction, sarcos, california_housing, airfoil_self_noise, yacht_hydrodynamics, concrete_slump_test_slump, concrete_slump_test_flow, concrete_slump_test_compressive_strength, computer_hardware, concrete_compressive_strength, csm, physicochemical_properties_of_protein, online_video_characteristics, kegg_metabolic_directed, communities_and_crime

## Every model excluding Saint and Node can run "fast" (less than 1 hour for all models):
christine, relative_location_of_ct, ujiindoorloc_latitude, ujiindoorloc_longitude, iris, wine, breast_cancer, adult, anneal, aps_failure, australian_credit, bank_marketing_old, blood_transfusion, car, cnae-9, connect-4, german_credit, dota2, htru2, insurance_company, internet_usage, kr-vs-kp, mfeat-fac, miniboone, mushroom, nomao, online_shoppers, qsar_bio, image_segmentation, seismic-bumps, shuttle, skin_segmentation, spambase, vehicle, walking, helena, jannis, jasmine, philippine, sylvine, volkert, amazon_employee, higgs_kaggle, blastchar, credit_card_fraud, numerai, shrutime, phoneme, kdd2009_appetency, kdd2009_churn, kdd2009_upselling, diabetes, year_prediction, sarcos, california_housing, airfoil_self_noise, yacht_hydrodynamics, concrete_slump_test_slump, concrete_slump_test_flow, concrete_slump_test_compressive_strength, computer_hardware, concrete_compressive_strength, csm, physicochemical_properties_of_protein, online_video_characteristics, kegg_metabolic_directed, communities_and_crime
