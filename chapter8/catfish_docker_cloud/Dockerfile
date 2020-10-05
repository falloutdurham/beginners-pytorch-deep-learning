FROM continuumio/miniconda3:latest

ARG port=5000

ENV CATFISH_PORT=$port

RUN conda install -y flask \
  && conda install -c pytorch torchvision \
  && conda install waitress
RUN mkdir -p /app

COPY ./catfish_model.py /app
COPY ./catfish_server.py /app
COPY ./run-model-service.sh /

EXPOSE $port

ENTRYPOINT ["/run-model-service.sh"]