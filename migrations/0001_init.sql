-- Registry, vector store and API keys.
--
-- {{EMBEDDING_DIM}} is substituted by the migration runner from
-- LOCAL_EMBEDDING_DIM. The dimension is part of the column type, not a value,
-- so it cannot drift from the model that produced the vectors without this
-- becoming a migration rather than a setting.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per knowledge base, written by both branches on success.
--
-- The context branch has no other home: its canvas existed only inside the
-- webhook body, so anything that wanted to answer a question from it later had
-- to intercept a message in flight. Storing it here is what makes a knowledge
-- base addressable after the job is gone.
CREATE TABLE IF NOT EXISTS knowledge_bases (
    id                BIGSERIAL PRIMARY KEY,
    kb_external_id    TEXT        NOT NULL UNIQUE,
    kb_type           TEXT        NOT NULL,
    branch            TEXT        NOT NULL CHECK (branch IN ('context', 'vector')),
    status            TEXT        NOT NULL DEFAULT 'ok',
    reason            TEXT,
    total_tokens      INTEGER     NOT NULL DEFAULT 0,
    -- Measured on the generated canvas, not the source documents: the fit check
    -- against the answering model's context window needs this number.
    canvas_tokens     INTEGER,
    estimator         TEXT,
    threshold         INTEGER,
    payload_id        BIGINT,
    job_id            BIGINT,
    -- Flowise store id, only when FLOWISE_ENABLED. null does not mean failure.
    document_store_id TEXT,
    item_count        INTEGER     NOT NULL DEFAULT 0,
    cost              JSONB       NOT NULL DEFAULT '{}'::jsonb,
    canvas            JSONB,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS knowledge_bases_branch_idx
    ON knowledge_bases (branch, status);

-- Retrievable fragments. Shared by the vector branch and by the context branch
-- when a canvas is split for sectioned retrieval, so one query path serves both.
CREATE TABLE IF NOT EXISTS chunks (
    id         BIGSERIAL PRIMARY KEY,
    kb_id      BIGINT      NOT NULL REFERENCES knowledge_bases (id) ON DELETE CASCADE,
    content    TEXT        NOT NULL,
    category   TEXT,
    source     TEXT,
    title      TEXT,
    metadata   JSONB       NOT NULL DEFAULT '{}'::jsonb,
    embedding  VECTOR({{EMBEDDING_DIM}}),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW over cosine is the index pgvector recommends; the composite kb_id column
-- in front of it is what keeps a query inside one knowledge base instead of
-- ranking the whole installation.
CREATE INDEX IF NOT EXISTS chunks_kb_id_idx ON chunks (kb_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Bearer keys for the query API. Only the SHA-256 of the key is stored, so a
-- leaked database row cannot be replayed against the API.
CREATE TABLE IF NOT EXISTS api_keys (
    id         BIGSERIAL PRIMARY KEY,
    name       TEXT        NOT NULL,
    key_hash   TEXT        NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS api_keys_active_idx
    ON api_keys (key_hash) WHERE revoked_at IS NULL;

-- Single row recording which model the stored vectors were produced with.
-- Managed by the migration runner, not by this file: the values come from
-- settings at migration time.
CREATE TABLE IF NOT EXISTS embedding_state (
    singleton  BOOLEAN    PRIMARY KEY DEFAULT true CHECK (singleton),
    model      TEXT       NOT NULL,
    dim        INTEGER    NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
