ARG port=8080

FROM continuumio/miniconda3:latest

ENV CATFISH_PORT=$port

RUN conda install -y flask && conda install -c pytorch torchvision && conda install -y requests pillow waitress
RUN mkdir -p /app

COPY ./catfish_model.py /app
COPY ./catfish_server.py /app
COPY ./run_model_service.sh /

EXPOSE $port

ENTRYPOINT ["/run_model_service.sh"]

