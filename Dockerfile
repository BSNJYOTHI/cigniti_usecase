FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8
WORKDIR /main
COPY . /main
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]