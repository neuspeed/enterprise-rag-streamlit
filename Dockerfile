FROM gitlab.infercom.one:5050/infercom-dev/llm-chains/knowledge-pipeline:vendor
ENV http_proxy http://10.3.51.17:8888
ENV https_proxy http://10.3.51.17:8888
ENV NO_PROXY=host.docker.internal,api.openai.com,registry-1.docker.io,10.2.50.202
ENV no_proxy=host.docker.internal,api.openai.com,registry-1.docker.io,10.2.50.202

WORKDIR /app

# Копируем только requirements сначала для кеширования
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app .

ENTRYPOINT ["python"]
CMD ["consumer_dispatcher.py"]