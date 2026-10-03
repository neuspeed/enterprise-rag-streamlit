from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    # Настройки инфраструктуры
    ENV: str = "dev"
    STAGE_BASIC_AUTH_USER: str = "pgadmin_stage_infercom"
    STAGE_BASIC_AUTH_PASS: str = "@1I6GiD7Ge-EMC]G%IHXETs92#!~f)UeP2mRzKQ6"
    PROXY: str | None = "10.3.51.17:8888"
    
    # Настройки для подключения к RabbitMQ
    MQ_HOST: str = ""
    MQ_PORT: int | None = None
    MQ_USER: str = ""
    MQ_PASS: str = ""
    MQ_VIRTUAL_HOST: str = "rabbit-vh"
    MQ_EXCHANGE: str = "{ENV}_knowledge_base_tasks"
    
    # DLQ настройки
    MQ_DLX_EXCHANGE: str = "dlx.knowledge.{ENV}"
    MQ_DLQ_QUEUE: str = "{ENV}_knowledge_base_tasks.dlq"
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: int = 60  # базовая задержка
    
    # Конкуретность
    GLOBAL_CONCURRENCY: int = 10
    CLIENT_WEIGHTS: dict[str,int] = {} # JSON вида {"client_a": 1, "client_b": 2}

    # Таймауты
    SHUTDOWN_DRAIN_TIMEOUT_SECONDS: int = 25
    MESSAGE_PROCESS_TIMEOUT_SECONDS: int = 1200
        
    # Настройка доступа к базе данных
    DATABASE: str = ""
    
    # Настройки для доступа к OwnCloud
    OWNCLOUD_BASE_URL: str = "https://cloud.infercom.one/remote.php/webdav/"
    OWNCLOUD_USER: str = "inferservice"
    OWNCLOUD_PASSWORD: str = "xIbektpFWMs8rAlE"
    LOCAL_PATH: str = "../tmp"  # Локальная папка для временных файлов
    
    # Настройки для подключения к локальной VLM модели (парсинг pdf)
    VLM_API_PROVIDER: str = "Infercom"  # Провайдер для локальной модели
    VLM_API_URL: str = "http://10.2.50.202:8088/v1"  # URL локальной модели
    VLM_API_KEY: str = "not-needed"
    VLM_MODEL_NAME : str = "gemma-4-31B-q8-it-GGUF"  # Имя модели для локальной модели
    VLM_PROMPT_TEMPLATE: str = """You are a helpful assistant that extracts structured information from documents. Extract the content and divide it into logical chunks. Return a JSON ARRAY where each object has: {{"content": "extracted text", "category": "document_chunk", "title": "optional header", "metadata": {{}}}}. ALWAYS return an array, even for a single chunk. DO NOT wrap in markdown, DO NOT include explanations or any text before/after the JSON. The response must be valid parseable JSON. Schema: {schema}"""
    VLM_TIMEOUT: int = 1800
    
    # Параметры для обработки pdf
    MAX_CONCURRENT_PDFS: int = 4
    POPPLER_PATH: str = "/usr/bin"
    
    
    # Настройки ветвления
    MAX_CKB_TOKENS: int = 220000 # Максимальное количество токенов для контекстной базы знаний (Context Knowledge Base)
    
    # Настройки для контекстной БЗ
    ENRICHER_API_URL: str = "https://a5000srv1.infercom.one/v1"
    ENRICHER_API_KEY: str = "not-needed"
    ENRICHER_MODEL_NAME: str = "gpt-oss-120bs"
    ENRICHER_PROMPT: str = '{ "chain_of_thought_requirement": "All internal chain-of-thought reasoning must be in English. Think step by step and follow the logic of the task to make decisions on the structure of the tasks described.", "role": "Senior prompt engineer with 20 years experience in preparing databases and knowledge bases for reasoning AI models. You are reading a knowledge base document that needs to be reworked into a JSON knowledge base for AI.", "objectives": [ "Analyse the content and determine the best structure for storing it in the JSON format of an AI knowledge base. Group the information in the most logical order, taking into account all the nuances of the knowledge provided.", "You can compose information in a logical way by grouping the data for a consistent and logical presentation. When using the language model, the AI must respond flexibly to this knowledge base, without errors and without loss of details", "Make sure that the information is presented in an optimal way, without redundant code.", "You can use all the data uploaded to you to form an optimal and comprehensive knowledge base.", "In cases where knowledge conflicts with each other and claims exclusionary concepts, you need to contextualize the variations in which they can be applied and not contradict each other.", "You can not shorten the meanings, you need to carefully ensure that the essence of knowledge is conveyed as in the original data." ], "output_instructions": [ "Return the knowledge base as a single, well‑formed JSON object. Do not wrap the output in any markdown code fences or other formatting.", "Do not impose length limits; include all relevant content to preserve meaning and key points.", "Retain product names, company names, abbreviations, and acronyms exactly as they appear in the source data.", "Include every URL and image URL in full, unchanged, and associate each clearly with the element it references within the JSON structure.", "Create a distinct "glossary" section that lists all specialized terms and their definitions in the original language, so the assistant can reference terminology contextually.", "Structure the JSON to separate main content, metadata, and the glossary, ensuring logical grouping and easy navigation for downstream AI processing.", "Avoid redundant code or unnecessary duplication; each piece of information should appear only once in its appropriate section." ] }'

    # Настройки для классификатора
    CLASSIFIER_API_URL: str = "http://10.2.50.202:8088/v1"
    CLASSIFIER_API_KEY: str = "not-needed"
    CLASSIFIER_MODEL_NAME: str = "google:gemma-4-e2b"
    CLASSIFIER_PROMPT_TEMPLATE: str = 'You are a text quality analyzer for a knowledge base. Evaluate the following fragment. Fragment: {chunk_text} Tasks: 1. Determine if the text contains useful information (is_useful: true/false). Useful information includes any meaningful data: facts, instructions, questions and answers, terms with definitions, lists, tables, narratives. Garbage includes navigation elements, advertisements, empty strings, technical noise. 2. If the text is useful, specify its structural type (category) strictly from the list below: {category_descriptions} 3. Estimate your confidence (confidence) from 0.0 to 1.0. Output must be strictly in JSON format without additional commentary: {{ "is_useful": true/false, "category": "faq", "confidence": 0.95 }} If the fragment is not useful, category can be set to "document_chunk", the main thing is is_useful=false.'
    CLASSIFIER_CATEGORY_DESCRIPTIONS: dict = {
        "faq": "explicit question-answer pair (e.g., 'Question: ... Answer: ...')",
        "glossary": "term and its definition",
        "document_chunk": "regular text (instruction, description, paragraph)",
        "table": "table (rows and columns)",
        "list": "bulleted or numbered list",
        "code": "code block or commands"
    }
    CLASSIFIER_TRASH_SIZE: int = 0
    
    # Настройки для структурирования данных
    LLM_STRUCT: str = ""
    
    # Настройки интеграции с Flowise
    FLOWISE_API_URL: str = "http://localhost:3000/api/v1"
    FLOWISE_API_KEY: str = "7y6g9GJ1tP361_13YVbrF9N1gyQG5Y8zHTtTl9BzQSc"
    
    # Настройки Flowise Document Store Upsert
    LOADER_NAME: str = "jsonFile"
    LOADER_CONFIG: dict = {}
    
    EMBEDDING_PROVIDER: str = "huggingFaceInferenceEmbeddings"
    EMBEDDING_API_URL: str = "http://10.2.50.202:8089"
    EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-8B"
    EMBEDDING_CREDENTIAL: str = "598c0f1b-2596-4ac2-9135-5354b7f44ba1"
    
    VECTOR_STORE_PROVIDER: str = "postgres"
    VECTOR_STORE_HOST: str = "host.docker.internal"
    VECTOR_STORE_PORT: int = 5432
    VECTOR_STORE_CREDENTIAL: str = "4a7bfa52-7c45-4414-bdb4-15c1d1ec2f23"
    VECTOR_STORE_DATABASE: str = "vector_db"
    VECTOR_STORE_TOPK: int = 10
    
    RECORD_MANAGER_PROVIDER: str = "postgres"
    RECORD_MANAGER_HOST: str = "host.docker.internal"
    RECORD_MANAGER_PORT: int = 5432
    RECORD_MANAGER_CREDENTIAL: str = "4a7bfa52-7c45-4414-bdb4-15c1d1ec2f23"
    RECORD_MANAGER_DATABASE: str = "records_db"