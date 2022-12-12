$DEV_PORT = 5002

try {
  docker stop DEV_inference_app_container
  docker rm DEV_inference_app_container
}
catch {}

$model_version = Read-Host -Prompt 'Specify model version to test'
docker build -t inference_app:$model_version -f ./dockerfiles/Dockerfile_serving . --build-arg MODEL_VERSION=$model_version
docker run -dit -p ${DEV_PORT}:${DEV_PORT} --name DEV_inference_app_container inference_app:$model_version
docker exec -d DEV_inference_app_container python -u model_serve_app.py DEV
python src/serve_app_utils/front_end_html_composer.py DEV
# Step 3: Invoke front end tool to prompt the user to do some testing
Invoke-Expression .\DEV_front_end.html
