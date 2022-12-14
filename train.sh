#!/usr/bin/env bash

read -p 'Specify model version to train: ' model_version

docker build -t delay_train_app:0.0.1 -f ./dockerfiles/Dockerfile_training . --build-arg MODEL_VERSION=$model_version
docker run -dit --name train_app_container delay_train_app:0.0.1
docker exec train_app_container python data_onboarding_main.py
docker exec train_app_container python preprocessing_main.py
docker exec train_app_container python train_main.py $model_version
# Bring data to the local machine
docker cp train_app_container:/delay_train_app/data/onboarded/ data/onboarded/
docker cp train_app_container:/delay_train_app/data/clean/ data/clean/
docker cp train_app_container:/delay_train_app/model_registry/$model_version/trained_model.joblib model_registry/$model_version
docker stop train_app_container
docker rm train_app_container

