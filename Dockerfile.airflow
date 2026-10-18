FROM apache/airflow:2.10.5-python3.12

USER airflow

# Install project dependencies into Airflow's Python environment
COPY --chown=airflow:root requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set default working directory
WORKDIR /opt/airflow
