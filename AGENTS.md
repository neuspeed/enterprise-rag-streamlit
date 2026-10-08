# Task for AI Agent (Opencode): Implement Knowledge Pipeline with Dynamic Branching & Streamlit UI

## 1. Context & Objective
You are an expert Python/AI platform engineer. Your task is to refactor and implement a modular data processing pipeline (`Knowledge Pipeline`) based on the provided architecture. 

The pipeline ingests raw messages from **RabbitMQ** or a **Streamlit Web UI**, processes various document types, automatically estimates tokens, splits execution into **two distinct architectural branches** based on a size/type threshold, and pushes the final structured JSON results into **Flowise Document Store via API / Webhook**.

The architecture must be fully asynchronous (`asyncio`), containerized via Docker, modular, robust to runtime failures, and follow Strict OOP/SOLID principles.

---

## 2. Project Architecture & Directory Structure
Implement the system according to the following layout:

```text
app/
├── ui/                 # Streamlit Web Interface
│   └── app.py
├── loaders/            # Cloud & local file loaders
├── cleaners/           # Parsers, text cleaners, Markdown normalization
├── splitters/          # Chunking modules (Large documents only)
├── classifier/         # LLM classification & filtering (Large documents only)
├── structurizers/      # QA & Glossary extractors (Large documents only)
├── enricher/           # High-density context canvas generation (Small documents only)
│   └── knowledge_canvas.py
├── pipeline.py         # Main execution coordinator & branching logic
├── config.py           # Environment variables configuration
├── utils/              # Data models, Flowise & RabbitMQ clients
│   ├── models.py
│   ├── flowise.py
│   └── rmq.py
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 3. Data Contracts & Domain Models (`app/utils/models.py`)
Implement strict data models using `pydantic` (v2) to prevent data drift between pipeline components.

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class RawElement(BaseModel):
    content: str = Field(..., description="Markdown or text content of the chunk/document")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Source URL, date, category (table, FAQ, glossary, document), etc."
    )

class PipelineMessage(BaseModel):
    kb_external_id: str
    kb_type: str = "auto"  # auto, force_small, force_large
    files: Optional[List[str]] = None  # Source URLs from the buyer's own storage
    html: Optional[List[str]] = None
    text: Optional[str] = None
```

---

## 4. Pipeline Execution Modules & Logic

### 4.1 Loaders & Parsers (`app/loaders/`, `app/cleaners/`)
* **Loaders**: Create an abstract `BaseLoader`. Implement support for downloading files (`.pdf`, `.docx`, `.xlsx`, `.csv`, `.txt`, `.html`) from the configured WebDAV mount or any public URL.
* **Cleaners & Parsers**: Extract raw data and normalize it into strict **Markdown** format with initial metadata (`RawElement`).
  * **PDF Parsing**: Implement parsing via a local **Gemma-4** model instance (mock or implement an actual LLM inference call as configured in `config.py`).
  * **HTML/Text Cleaners**: Strip technical noise (tags, CSS, wrappers) and keep core layout/tables intact.

### 4.2 Token Estimation & Dynamic Branching (`app/pipeline.py`)
* Use a universal tokenizer (e.g., `tiktoken` or standard LLM context tokenizer based on `config.py`).
* Read `CONTEXT_WINDOW_THRESHOLD` (default: 0.70 of max context window) from config.
* **Execution Paths**:
  * If `total_tokens <= threshold` OR `kb_type == 'force_small'` → Execute **Small Document Branch**.
  * If `total_tokens > threshold` OR `kb_type == 'force_large'` → Execute **Large Document Branch**.

### 4.3 Small Document Branch (`app/enricher/`)
* **Module**: `app/enricher/knowledge_canvas.py`
* **Action**: Do NOT chunk the document. Invoke the LLM API using the `Enricher` prompt: *"Создай контекстное полотно знаний на основе документа..."*
* **Output**: High-density structured context text/JSON wrapped into a final `RawElement`.

### 4.4 Large Document Branch (`app/splitters/`, `app/classifier/`, `app/structurizers/`)
* **Splitter**: Chop the main text into a `List[RawElement]` using a configured chunk size and logical overlap.
* **Classifier**: Filter out uninformative chunks/garbage using a lightweight LLM prompt or minimum character length constraint.
* **Structurizer**: Extract explicit QA pairs and glossary definitions from the remaining valid chunks using targeted LLM system prompts and regular expressions.

---

## 5. Flowise Integration & Naming Convention (`app/utils/flowise.py`)

Every Knowledge Base must be routed deterministically to its dedicated Document Store using a standard `kb_` prefix. Implement the following logic asynchronously (`httpx` or `aiohttp`):

```python
KB_NAME_PREFIX = "kb_"

async def get_or_create_document_store(flowise_client, kb_external_id: str) -> str:
    """
    1. Construct deterministic store name: f"{KB_NAME_PREFIX}{kb_external_id}"
    2. Query Flowise API to check if this store name already exists.
    3. If exists: return its document_store_id.
    4. If not: execute Flowise API creation endpoint, create the store, and return the new ID.
    """
    pass
```
* **Output Formatter**: Assemble the final array of structured JSON tokens/metadata, push them into Flowise Document Store via SDK/API, and ping the **KB API Webhook** with the sync result.

---

## 6. Streamlit User Interface (`app/ui/app.py`)
Implement a lightweight developer and testing UI dashboard that directly imports pipeline modules:
1. **Sandbox Document Upload**: Support manual file drag-and-drop (`.pdf`, `.docx`, `.txt`).
2. **Pipeline State Visualizer**:
   * Count and print total tokens before running.
   * Highlight which execution branch is triggered (**Small** vs **Large**).
3. **Payload Preview**: Render the generated output JSON and structural metadata directly on-screen before pushing it to the target database.
4. **Target Store Targeting**: Manual `kb_external_id` text field to test deterministic store creation and routing.

---

## 7. Multi-Container Infrastructure (`docker-compose.yml`)
Configure the multi-container stack to keep the processing worker independent from the web UI layer while sharing system configs and volumes.

```yaml
version: '3.8'

services:
  pipeline_worker:
    build: .
    container_name: pipeline_worker
    command: python app/pipeline.py
    environment:
      - LLM_API_KEY=${LLM_API_KEY}
      - LLM_API_URL=${LLM_API_URL}
      - FLOWISE_API_URL=${FLOWISE_API_URL}
      - RMQ_HOST=rabbitmq
    volumes:
      - .:/app
    depends_on:
      - rabbitmq

  streamlit_ui:
    build: .
    container_name: pipeline_ui
    command: streamlit run app/ui/app.py --server.port=8501 --server.address=0.0.0.0
    ports:
      - "8501:8501"
    environment:
      - LLM_API_KEY=${LLM_API_KEY}
      - LLM_API_URL=${LLM_API_URL}
      - FLOWISE_API_URL=${FLOWISE_API_URL}
    volumes:
      - .:/app

  rabbitmq:
    image: rabbitmq:3-management
    ports:
      - "5672:5672"
      - "15672:15672"
```

## 8. Requirements for Code Generation
1. Write idiomatic, readable, clean Python 3.10+ code.
2. Use fully async network calls (`httpx.AsyncClient` or `aiohttp`) for cloud storage, LLM APIs, and Flowise endpoints.
3. Ensure the code reads configuration dynamically from standard environment variables via `app/config.py`.

Generate all module skeletons and the core execution logic now.
