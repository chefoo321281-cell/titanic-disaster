FROM python:3.11-slim

WORKDIR /app

# install packages first so docker can cache this step
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY titanic_model.py .

# the script looks for train.csv / test.csv here (we mount src/data into this folder when running)
ENV DATA_DIR=/app/data

CMD ["python", "titanic_model.py"]
