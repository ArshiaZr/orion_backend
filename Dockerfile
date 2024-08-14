FROM python:3.9-slim-buster

# Set up work directory
WORKDIR /MLH-portfolio-website

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]

EXPOSE 5000