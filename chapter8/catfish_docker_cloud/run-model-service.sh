#!/bin/bash
#run-model-service.sh

cd /app
waitress-serve --port ${CATFISH_PORT} --call 'catfish_server:create_app'
