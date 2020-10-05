#!/bin/bash
#run-docker.sh

docker build -t catfish-service .
docker run -d -p 5000:5000  catfish-service:latest
