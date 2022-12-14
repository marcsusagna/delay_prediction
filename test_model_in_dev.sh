#!/usr/bin/env bash

DEV_PORT=5002

# Stop and remove the DEV_inference_app_container container if it exists
docker stop DEV_inference_app_container 2>/dev/null
docker rm DEV_inference_app_container 2>/dev/null

read -p 'Specify model version to test: ' model_version

docker build -t inference_app:$model_version -f ./dockerfiles/Dockerfile_serving . --build-arg MODEL_VERSION=$model_version
docker run -dit -p $DEV_PORT:$DEV_PORT --name DEV_inference_app_container inference_app:$model_version
docker exec -d DEV_inference_app_container python -u model_serve_app.py DEV $model_version
python src/serve_app_utils/front_end_html_composer.py DEV
# Step 3: Invoke front end tool to prompt the user to do some testing
xdg-open DEV_front_end.html 2>/dev/null

