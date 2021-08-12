FROM continuumio/miniconda:4.5.4

COPY requirements.txt .

RUN mkdir /app
COPY ./app
RUN requirements.txt

CMD [ "python", "server.py" ]