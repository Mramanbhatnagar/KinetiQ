import os
import sys
import logging
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer

# 1. Pipeline Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "ai_feature_store")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")

# Initialize a production-standard local transformer model (384 Dimensions)
logging.info("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_connection():
    return psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)

def seed_raw_data():
    """Seeds sample documents mimicking production technical articles."""
    conn = get_connection()
    cur = conn.cursor()
    
    sample_docs = [
        ("PostgreSQL Indexing", "Database performance relies heavily on B-Tree and GIN indexes for fast queries.", "Engineering"),
        ("Transformers and LLMs", "Attention mechanisms in transformer networks allow deep learning models to capture long-range text dependencies.", "AI/ML"),
        ("Vector Databases for RAG", "Retrieval Augmented Generation systems use dense vector embeddings to look up relevant contexts from databases.", "AI/ML")
    ]
    
    try:
        cur.execute("DELETE FROM raw_documents;")
        for title, content, cat in sample_docs:
            cur.execute(
                "INSERT INTO raw_documents (title, content, category) VALUES (%s, %s, %s) RETURNING doc_id;",
                (title, content, cat)
            )
        conn.commit()
        logging.info("Successfully seeded raw text documents.")
    except Exception as e:
        conn.rollback()
        logging.error(f"Seeding failed: {e}")
    finally:
        cur.close()
        conn.close()

def generate_and_store_embeddings():
    """Chunks text, runs AI model inference, and saves embeddings into the SQL store."""
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("SELECT doc_id, content FROM raw_documents;")
        docs = cur.fetchall()
        
        cur.execute("DELETE FROM document_embeddings;")
        
        for doc_id, content in docs:
            # AI Logic: Chunking simulation (Simple clean sentences)
            chunks = [sentence.strip() for sentence in content.split('.') if len(sentence.strip()) > 5]
            
            for chunk in chunks:
                # Generate AI Vector Embedding
                embedding = model.encode(chunk).tolist()
                token_estimate = len(chunk.split())
                
                cur.execute(
                    """
                    INSERT INTO document_embeddings (doc_id, chunk_text, vector_data, token_count)
                    VALUES (%s, %s, %s, %s);
                    """,
                    (doc_id, chunk, embedding, token_estimate)
                )
        conn.commit()
        logging.info("AI Vectors calculated and successfully stored in the database.")
    except Exception as e:
        conn.rollback()
        logging.error(f"Embedding generation failed: {e}")
    finally:
        cur.close()
        conn.close()

def execute_semantic_search(query_text, top_k=2):
    """Executes a pure mathematical Cosine Similarity lookup natively inside SQL."""
    conn = get_connection()
    cur = conn.cursor()
    
    # Encode user question into the exact same AI vector space
    query_vector = model.encode(query_text).tolist()
    
    # Advanced Vector Math Query: Cosine Similarity calculated natively via dot product and norms
    search_sql = """
        SELECT 
            rd.title,
            de.chunk_text,
            rd.category,
            (
                SELECT SUM(a * b)
                FROM UNNEST(de.vector_data, %s) AS x(a, b)
            ) / (
                SQRT((SELECT SUM(a * a) FROM UNNEST(de.vector_data) AS x(a))) * 
                SQRT((SELECT SUM(b * b) FROM UNNEST(%s) AS y(b)))
            ) AS cosine_similarity
        FROM document_embeddings de
        JOIN raw_documents rd ON de.doc_id = rd.doc_id
        ORDER BY cosine_similarity DESC
        LIMIT %s;
    """
    
    try:
        cur.execute(search_sql, (query_vector, query_vector, top_k))
        results = cur.fetchall()
        
        print(f"\n--- AI SEMANTIC SEARCH RESULTS FOR: '{query_text}' ---")
        for i, row in enumerate(results):
            print(f"Rank {i+1} [Sim: {row[3]:.4f}] | Source: {row[0]} ({row[2]}) -> \"{row[1]}\"")
    except Exception as e:
        logging.error(f"Search query failed: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    seed_raw_data()
    generate_and_store_embeddings()
    
    # Test semantic retrieval capability
    execute_semantic_search("How do RAG applications retrieve information context?")
    execute_semantic_search("Optimizing structured database query systems.")
