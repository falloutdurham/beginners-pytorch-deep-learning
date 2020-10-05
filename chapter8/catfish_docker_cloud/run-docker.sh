#!/bin/bash
#run-docker.sh

docker build -t catfish-service .
docker run -d -p 5000:5000 --env CATFISH_MODEL_LOCATION=[URL] catfish-service:latest