FROM nvidia/cuda:11.2.2-devel-ubuntu18.04

RUN apt update && apt install -y git python3 python3-pip supervisor
RUN pip3 install --upgrade pip
RUN python3 -m pip install jupyter

ENTRYPOINT ["supervisor"]