#!/bin/bash

PROD_PORT=5000
TEST_PORT=5001

# Continuous Integration (CI): Nothing to do bc we run on master and we don't have test
# Continuous Delivery (CD): Create docker image for model serving
# Step 1: Get current model version:
model_version=$(python src/model/get_current_model_version.py)
docker build -t inference_app:$model_version -f ./dockerfiles/Dockerfile_serving . --build-arg MODEL_VERSION=$model_version

# Continuous Deployment (CD): 
# Step 1: Spin up docker container in TEST environment
docker run -dit -p ${TEST_PORT}:${TEST_PORT} --name TEST_inference_app_container inference_app:$model_version
docker exec -d TEST_inference_app_container python -u model_serve_app.py TEST $model_version

# Step 2: Create test environment entry point
python src/serve_app_utils/front_end_html_composer.py TEST

# Step 3: Invoke front end tool to prompt the user to do some testing
./TEST_front_end.html

# Step 4: Ask the user to confirm if acceptance test has passed and we can deploy to PROD
read -p "Can we deploy to PROD? Reply with Y: " deploy_to_prod
if [ "$deploy_to_prod" = "Y" ]; then
  docker stop PROD_inference_app_container
  docker rm PROD_inference_app_container
  docker run -dit -p ${PROD_PORT}:${PROD_PORT} --name PROD_inference_app_container inference_app:$model_version
  docker exec -d PROD_inference_app_container python -u model_serve_app.py PROD $model_version
  python src/serve_app_utils/front_end_html_composer.py PROD

  # Open the generated HTML file
  xdg-open PROD_front_end.html
fi

# Clean up the TEST environment
rm TEST_front_end.html
docker stop TEST_inference_app_container
docker rm TEST_inference_app_container



