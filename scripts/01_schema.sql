-- Clean setup for validation
DROP TABLE IF EXISTS document_embeddings CASCADE;
DROP TABLE IF EXISTS raw_documents CASCADE;

-- 1. Raw Text Ingestion Table
CREATE TABLE raw_documents (
    doc_id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. AI Vector Feature Table (Using raw arrays to run anywhere out-of-the-box)
CREATE TABLE document_embeddings (
    embedding_id SERIAL PRIMARY KEY,
    doc_id INT REFERENCES raw_documents(doc_id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    -- 384-dimensional dense vector representing the text semantics
    vector_data REAL[] NOT NULL, 
    token_count INT NOT NULL,
    CONSTRAINT check_vector_dimensions CHECK (array_length(vector_data, 1) = 384)
);

-- 3. Optimize Text Fields for traditional Keyword Search (Enables Hybrid Search)
CREATE INDEX idx_raw_docs_content_tsvector ON raw_documents USING gin(to_tsvector('english', content));
