# KinetiQ
AI High-Performance Vector Search & RAG Feature Store. 
# Project Description
This repository implements an end-to-end semantic search and data preprocessing pipeline designed for modern Generative AI and RAG architectures. It bridges unstructured text data with a highly optimized vector-capable database backend to deliver rapid contextual search results.
# Key Production-Grade AI Features
    Automated Text Chunking & Embedding: Utilizes a lightweight, open-source Transformer model (all-MiniLM-L6-v2) to convert unstructured articles into 384-dimensional dense vectors.
    
    Cosime Similarity Engine: Uses mathematically rigorous cosine distance formulas directly inside PostgreSQL to rank document relevance without external heavy infrastructure.
    
    Hybrid Keyword-Semantic Retrieval: Combines traditional SQL full-text search indexes with mathematical vector scoring for a robust hybrid search framework.
    
    Idempotent Data Validation: Built-in SQL constraints and upsert mechanisms to completely avoid embedding drift and dataset duplication.
# Database Schema Definition

├── .github/

│   └── workflows/

│       └── ai-test-pipeline.yml

├── scripts/

│   └── 01_schema.sql

├── ai_pipeline.py

├── requirements.txt

└── README.md
