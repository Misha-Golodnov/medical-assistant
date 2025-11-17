-- Инициализация базы данных Medical Assistant
-- Автоматически выполняется при первом запуске контейнера

-- Создаем расширение pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Создаем таблицу метаданных клинических рекомендаций
CREATE TABLE IF NOT EXISTS cr_metadata (
    id SERIAL PRIMARY KEY,
    cr_id VARCHAR(50) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    icd_codes TEXT[],
    url TEXT,
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Создаем таблицу чанков с векторными embeddings
-- OpenAI text-embedding-3-large использует 3072 измерения
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    cr_id VARCHAR(50) NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding vector(3072),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cr_id) REFERENCES cr_metadata(cr_id) ON DELETE CASCADE
);

-- Создаем индексы для быстрого поиска
-- Примечание: Векторный индекс будет создан позже с правильными параметрами
-- pgvector в PostgreSQL 16 ограничен 2000 измерениями для индексов
-- Мы используем text-embedding-3-large (3072 dim), поэтому создадим индекс без ограничений
-- или будем использовать точный поиск (медленнее, но работает)

-- Индексы для метаданных
CREATE INDEX IF NOT EXISTS chunks_cr_id_idx ON chunks(cr_id);
CREATE INDEX IF NOT EXISTS cr_metadata_icd_codes_idx ON cr_metadata USING GIN(icd_codes);

-- Создаем таблицу для логов запросов
CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    user_id BIGINT,
    username VARCHAR(255),
    n_results INTEGER,
    similarity_threshold REAL,
    top_similarity REAL,
    results_count INTEGER,
    context_chars INTEGER,
    context_crs INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    embedding_cost NUMERIC(10, 6),
    input_cost NUMERIC(10, 6),
    output_cost NUMERIC(10, 6),
    total_cost NUMERIC(10, 6),
    cached BOOLEAN DEFAULT FALSE,
    response_length INTEGER,
    execution_time_ms INTEGER,
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Индексы для анализа логов
CREATE INDEX IF NOT EXISTS query_logs_created_at_idx ON query_logs(created_at);
CREATE INDEX IF NOT EXISTS query_logs_user_id_idx ON query_logs(user_id);
CREATE INDEX IF NOT EXISTS query_logs_cached_idx ON query_logs(cached);

-- Выводим информацию о созданных объектах
DO $$
BEGIN
    RAISE NOTICE 'Database initialized successfully!';
    RAISE NOTICE 'Tables created: cr_metadata, chunks, query_logs';
    RAISE NOTICE 'Indexes created: chunks_cr_id_idx, cr_metadata_icd_codes_idx, query_logs_*';
    RAISE NOTICE 'Extension enabled: vector';
END $$;
