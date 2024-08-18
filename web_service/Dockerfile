# Use the slim Python 3.10 base image
FROM python:3.10-slim

# Upgrade pip and install pipenv
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir pipenv

# Set the working directory
WORKDIR /app

# Copy the Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install the Python dependencies using pipenv
RUN pipenv install --deploy --ignore-pipfile

# Copy the application files
COPY predict.py utils.py ./

COPY .env .env

# Expose the port for the application
EXPOSE 9696

# Run the application using gunicorn via pipenv
CMD ["pipenv", "run", "gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
