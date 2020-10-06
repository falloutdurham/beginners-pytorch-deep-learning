FROM continuumio/miniconda3:latest

ARG model_parameter_file=catfishweights.pth
ARG port=5000

ENV CATFISH_PORT=$port
ENV CATFISH_MODEL_LOCATION=/app/$model_parameter_file

RUN conda install -y flask \
  && conda install -c pytorch torchvision \
  && conda install waitress
RUN mkdir -p /app

COPY ./catfish_model.py /app
COPY ./catfish_server.py /app
COPY ./$model_parameter_file /app/
COPY ./run-model-service.sh /

EXPOSE $port

ENTRYPOINT ["/run-model-service.sh"]