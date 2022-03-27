FROM ubuntu:20.04

RUN apt-get update
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install python3.9 -y
RUN apt-get install python3-pip -y
RUN apt install python3.8-venv

COPY requirements.txt boot.sh flask_main.py ./
COPY src src
COPY static static
RUN python3 -m venv venv
RUN venv/bin/pip install -r requirements.txt
RUN venv/bin/pip install gunicorn

RUN chmod +x boot.sh

EXPOSE 5000
ENTRYPOINT ["./boot.sh"]