## syntax=docker/dockerfile:1

FROM python:3.8-slim-buster
FROM ubuntu:18.04

WORKDIR /app

FROM ubuntu:latest
RUN apt-get -y update
RUN apt-get -y install git

RUN apt-get update && apt-get install -y \
curl
CMD /bin/bash
RUN apt install build-essential


RUN apt-get update && \
    apt-get install -y -q --allow-unauthenticated \
    git \
    sudo
RUN useradd -m -s /bin/bash linuxbrew && \
    usermod -aG sudo linuxbrew &&  \
    mkdir -p /home/linuxbrew/.linuxbrew && \
    chown -R linuxbrew: /home/linuxbrew/.linuxbrew
USER linuxbrew
RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"


RUN echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> /home/linuxbrew/.profile
RUN echo eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"




RUN /home/linuxbrew/.linuxbrew/bin/brew update
RUN /home/linuxbrew/.linuxbrew/bin/brew install gcc
RUN /home/linuxbrew/.linuxbrew/bin/brew install pkg-config



RUN /home/linuxbrew/.linuxbrew/bin/brew install homebrew/science/ipopt
RUN /home/linuxbrew/.linuxbrew/bin/brew install glpk


FROM python:3.8-slim-buster
COPY requirements2.txt requirements2.txt
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements2.txt
RUN pip install -r requirements.txt
COPY . .

CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]


CMD ["test.py"]
ENTRYPOINT ["python"]


