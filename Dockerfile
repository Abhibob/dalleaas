FROM python:3.11.1-slim-bullseye
RUN apt-get update && \
    apt-get install -y git
RUN pip install uuid jax dalle_mini transformers flask
RUN pip install git+https://github.com/patil-suraj/vqgan-jax.git
WORKDIR /usr/src/app
COPY . .
ENV FLASK_APP /usr/src/app/src/inference.py
CMD ["flask", "run", "--host=0.0.0.0"]