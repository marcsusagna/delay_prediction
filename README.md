## The model

### Objective
Explain target

### Data preparation and feature extraction (data pipeline)
Explain 

### Modeling

### Model evaluation rationale

ML vs delay analysis
Why on 2022 for 2021?? Do not overfit! otw we have seen these flights, don't overestimate performance

### How to predict for 2023

How to create contrafactual dataset

## Model development and evaluation

Explain model_selection_main.py and model_evaluation_main.py


## Model Lifecycle


### Deploying the last version of the model


### Deploying model training


### Model lifecycle: From new version developing to deployment

Steps to develop a new version of the model:
1. Branch off master
2. Go to src/model/constants and increase MODEL_NEW_VERSION
3. Do your changes (normally in feature extraction and/or model selection)
4. If you changed features, run again preprocessing_main.py
5. Run (manually) model_selection_main.py to create the new model's blueprint
6. run model_evaluation_main.py MODEL_NEW_VERSION to enhance blueprint with 2022 contrafactual prediction
7. Run on train_main.py MODEL_NEW_VERSION
8. Test model on a DEV environment by running on powershell test_model_in_dev.p1 with version MODEL_NEW_VERSION. You can compare to the PROD version by opening the PROD front-end in parallel. 
9. Check in the DEV front-end the model metrics / 2023 behavior
10. Increase MODEL_CURRENT_VERSION
11. Raise PR, merge to master
12. Run ci_cd.ps1 on powershell with the model version you just created. 
13. Do your tests in TEST environment. Sign off deployment to PROD by answering Y to the powershell prompt
14. Open PROD front-end to see new model was promoted to PROD. 

### Model retraining


### Model retraining because new data has arrived

## Further steps on how to improve the model:

Features: weather, airport capacity
Outlier handling

Evaluation: Randomness: on response variable scale 