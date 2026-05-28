FROM python:3.11-slim
WORKDIR /srv
COPY requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /srv
EXPOSE 3000
CMD ["streamlit", "run", "portal.py", "--server.port", "3000", "--server.address", "0.0.0.0"]
