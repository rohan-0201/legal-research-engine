# Overview
Legal professionals often spend hours reading through lengthy case documents to find relevant precedents and arguments. This project aims to streamline that process using an AI-powered legal research engine that retrieves and summarizes similar judgments based on user queries. By integrating semantic search and large language models, this system enhances legal reasoning, improves productivity, and supports efficient case law discovery.

The engine leverages vector-based retrieval and generative summarization to identify and explain relevant sections from judgments. It is particularly tuned for Australian Commercial Court cases, including domains like Taxation, Industrial & Labour, and Financial Disputes.

# Project Architecture
## Data Preprocessing
- Extracts and cleans judgment texts from Australian Legal XML Corpus.
- Filters cases belonging to relevant categories: Tax, Financial, and Industrial & Labour.
- Performs text normalization, stopword removal, lemmatization, and sentence segmentation.

## Embedding and Vector store
- Uses LegalBERT or sentence-transformers to generate embeddings.
- Stores vectors and corresponding case metadata in FAISS or ChromaDB.
- Each case is split into semantically meaningful sections for fine-grained retrieval.

## Query and Retrieval
- User inputs a legal query or a paragraph of a new case.
- The top-k most similar sections (e.g., k = 10) are retrieved based on cosine similarity.
- Retrieves both content and metadata (case name, section type, similarity score).

## Summarization
- Summarization of relevant sections is performed using Gemini 2.0 Flash API.
- Generates a concise explanation of each section and how it relates to the query.

# Components
## Preprocessing pipeline
- Parses XML judgments.
- Selects and cleans Commercial Court cases.
- Applies NLP techniques for text normalization.
- Outputs structured .json or .csv files with clean text and metadata.

## Embedding Generator
- Loads LegalBERT model or other transformer-based encoder.
- Embeds each case section.
- Stores section embeddings and references to full case metadata.

## Vector store and Retrieval
- Implements FAISS/ChromaDB index.
- Given a user query, retrieves top-k similar sections.
- Returns text, case name, section title, and similarity score.

## Summarization Engine
- Uses Google Gemini 2.0 Flash for summarization.
- Summarizes retrieved sections to explain legal relevance.
- Ensures responses are case-specific and concise.

# Results
Using top-10 semantic retrieval, the system effectively surfaces the most relevant case sections. Gemini 2.0 Flash provides concise summaries, enabling legal researchers to understand case relevance without reading full documents. This results in faster legal insight, reduced manual search, and improved decision support for lawyers and researchers.
