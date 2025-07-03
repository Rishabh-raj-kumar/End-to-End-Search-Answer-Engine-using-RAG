# End-to-End Search & Answer Engine using RAG

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline that transforms any PDF document into an interactive and intelligent question-answering system. By leveraging state-of-the-art sentence embeddings and large language models (LLMs), this engine can understand natural language queries, find the most relevant information within a document, and generate accurate, context-aware answers.


## Key Features

- **Document Processing:** Efficiently chunks text from large PDF documents for granular analysis.
- **Semantic Search:** Uses high-quality sentence embeddings (all-mpnet-base-v2) to find the most semantically relevant passages to a user's query, going beyond simple keyword matching.
- **Intelligent Answer Generation:** Employs Google's powerful Gemma LLM (gemma-2b-it) to synthesize information from the retrieved context and generate human-like, accurate answers.
- **Optimized Prompting:** Implements a custom, sophisticated prompt template that guides the LLM to "think" before answering, drastically improving the quality and factuality of the generated text.
- **Modular & Scalable:** The pipeline is designed to be easily adaptable to different documents, embedding models, and language models.

## Tech Stack

- **Core Framework:** Python 3.x
- **Machine Learning:** PyTorch
- **NLP & LLMs:** Hugging Face (transformers, sentence-transformers)
- **Models Used:**
  - **Embedding Model:** all-mpnet-base-v2
  - **Language Model:** google/gemma-2b-it
- **Data Handling:** Pandas, NumPy
- **PDF Processing:** PyMuPDF

## How It Works

The project follows a three-stage RAG pipeline:

1. **Retrieval:** When a user asks a question, the query is first encoded into a vector embedding using the same model that processed the source document. A similarity search (dot product) is performed between the query embedding and the embeddings of all text chunks from the document. The top-K most similar chunks are retrieved.

2. **Augmentation:** The retrieved text chunks are then formatted and inserted into a custom prompt template. This template provides the LLM with the necessary context, examples of high-quality answers, and instructions to reason before responding.

3. **Generation:** The augmented prompt is sent to the Gemma language model, which generates a comprehensive and accurate answer based only on the provided context. This step ensures the answers are grounded in the source document, minimizing the risk of hallucination.

## Setup and Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required libraries:**

   ```bash
   pip install -r requirements.txt
   ```

   (Ensure you have a `requirements.txt` file with all the necessary packages like torch, transformers, sentence-transformers, pandas, numpy, pymupdf)

4. **Hugging Face Authentication (Required for Gemma):**

   You need to be authenticated with Hugging Face to download the Gemma model.

   ```bash
   huggingface-cli login
   ```

   Paste your Hugging Face access token when prompted.

