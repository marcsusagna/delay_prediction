## Task: 

Description: Predict delay in 2023 using the flight from 2022 schedule

Solution: 

- Set up a model in which the delay (y) is a function of x_demand and x_others, where x_demand takes somehow into account
the demand for that leg.
- Train a model on 2021 and 2022 data to predict y
- Create 2023 dataset based on 2022 legs but with an increase (user input) on x_demand feature (taking into account
aircraft capacity and hence, how much new clients could the airline absorbe)
- Provide a prediction for this contrafactual 2023 data. 

There are two possible business questions to answer: 
- Delay frequency: How many flights are delayed? We fit a classification model for this setting. 
- Delay time: How is the delay time distributed in 2023? This can help answers like number of delays with more
than 15 min delay (which has an implication on client satisfaction). We train a regression model for this case. 

## Model Lifecycle


### First deployment

Pre-steps:
- Install docker on your computer
- Clone the repository and make it's root as your working directory
- Create a conda environment and install the dependencies in requirements_local_exploration.txt in case you want to 
develop new model versions or run scripts locally. 
- Copy the data (delay.csv and fis.csv) into a folder called source within the data folder. Meaning the file paths
will be data/source/delay.csv and data/source/fis.csv

Now you can deploy the latest model by:
- Execute the file train.ps1 (or train.sh if on bash. Also execute with sudo!). Prompt the latest model version (v0.0.7)
- Execute the file ci_cd.ps1 You'll be prompted with a test environment. Enter Y in the powershell script to deploy to PROD
- You can access PROD by opening the newly created html file: PROD_front_end


### Deploying model training

Training can be run locally with:

$ python train_main.py {model_version}

Or you can train within a container by executing train.ps1

The reasons for having train as a runnable container is to deploy it to PROD environment and train there. This is needed if
- PROD data has more records or has confidental data (i.e. real data compared to dev env)
- Your local machine is too small and you want to run it on a stronger one remotely

### Model lifecycle: From new version developing to deployment

Steps to develop a new version of the model:
1. Branch off master
2. Go to src/model/constants and increase MODEL_NEW_VERSION
3. Do your changes (normally in feature extraction and/or model selection)
4. If you changed features, run again preprocessing_main.py. If new data onboarded, run data_onboarding_main.py
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

In the scenario in which PROD performance deteriorates we might want to retrain and deploy the model. You can do that by

- Running train_main.py
- Executing ci_cd.ps1

Also, in the case you need to deploy training (i.e. training in PROD or bigger machine) instead of running 

$ python train_main.py {model_version}

You can execute train.ps1

In a real production setup, one might want to have a schedule based retraining. The schedule triggers could be:

- Every x days
- As soon as a monitoring alert notifies of performance deterioration

Then the two commands above would be automatically run. 


### Model retraining because new data has arrived

As before, we would just need to:

- Execute train.ps1 (it contains the data pipeline)
- Execute ci_cd.ps1

Note that currently the repository assumes static data only. One would need to take care of the data preprocessing pipeline
for instance by limiting to two last years of data or adapt to whatever new data handling policy. 