# PalmMind RAG API

This project is a sophisticated, multi-faceted Retrieval-Augmented Generation (RAG) application built with FastAPI. It provides a robust backend for document analysis, a conversational chat interface with booking capabilities, and a comprehensive benchmarking suite to evaluate its own performance.

## Features

- **Conversational RAG API:** A powerful chat interface that can answer questions based on ingested documents and handle multi-turn conversations.
- **Intelligent Booking System:** A stateful, conversational booking system integrated into the chat, allowing users to schedule appointments by providing their name, email, date, and time.
- **LLM-Powered Intent Routing:** Uses a Gemini model with structured output to intelligently route user requests between greeting, retrieval, and booking modes based on the conversation history.
- **Document Ingestion API:** An endpoint to upload PDF or TXT files, which are then processed, chunked, and stored in a vector database.
- **Selectable Chunking Strategies:** Supports multiple document chunking strategies (`fixed_size`, `sentence_split`) that can be chosen at the time of ingestion.
- **Vector Database Integration:** Uses Pinecone as the vector store for efficient similarity search on document embeddings.
- **Comprehensive Benchmarking Suite:** Includes scripts to rigorously evaluate and compare the performance of different chunking strategies and similarity metrics.

---

## Project Structure

```
/home/xyz/Documents/PalmMind/
├───.env
├───src/
│   ├───api/            # FastAPI routers for different endpoints (chat, booking, ingestion).
│   ├───chunkers/       # Different text chunking strategy implementations.
│   ├───models/         # SQLAlchemy models for database tables (documents, bookings).
│   ├───services/       # Core business logic (RAG service, booking, document processing).
│   ├───myapp/          # Main FastAPI application setup and entry point.
│   └───...             # Other modules for database, clients, etc.
├───benchmarks/         # Scripts for performance evaluation.
│   ├───compare_chunkers.py
│   └───compare_similarity.py
├───tests/
│   └───test_data/      # Test documents and question-answer pairs for benchmarking.
├───requirements.txt    # Project dependencies.
└───README.md
```

---

## Core Logic

### Conversational RAG & Booking

The heart of the application is the `rag_service.py`, which uses `langgraph` to create a state machine that manages the conversational flow. 

1.  **Stateful Sessions:** User session history, including messages and in-progress booking details, is stored in Redis to maintain context across multiple interactions.
2.  **Intent Detection:** At the start of each turn, a Gemini model is used to analyze the conversation and determine the user's intent (`greeting`, `retrieval`, or `booking`).
3.  **Conditional Routing:** Based on the detected intent, the graph routes the conversation to the appropriate node:
    *   **`greeting`:** Provides a simple greeting.
    *   **`retrieve_context`:** Queries the Pinecone vector store for relevant document chunks.
    *   **`generate_response`:** Generates a response using the retrieved context.
    *   **`handle_booking`:** Guides the user through a multi-turn conversation to collect booking details, saves the booking to a SQL database, and sends a confirmation email.

### Benchmarking

The `benchmarks/` directory contains two key scripts for evaluation:

-   **`compare_chunkers.py`:** This script tests different text chunking strategies. For each strategy, it ingests a set of test documents and then evaluates retrieval performance using a ground-truth question-answer dataset. It calculates Accuracy, Recall, and Latency.
-   **`compare_similarity.py`:** This script compares different similarity metrics (e.g., `cosine` vs. `dotproduct`). For each metric, it creates a new Pinecone index, ingests documents, and runs the same evaluation to calculate Accuracy, Precision, Recall, F1-Score, and Latency.

---

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd PalmMind
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the root directory and add the following, replacing the placeholder values:
    ```
    # Pinecone
    PINECONE_API_KEY=YOUR_PINECONE_API_KEY
    PINECONE_ENVIRONMENT=your-pinecone-environment

    # Google AI
    GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY

    # SMTP for Email Confirmation
    SMTP_SERVER=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=your.email@gmail.com
    SMTP_PASSWORD=your-google-app-password 
    ```

---

## Running the Application

1.  **Start the Redis Server:**
    Ensure Redis is installed and running. If not, you can start it with:
    ```bash
    redis-server &
    ```

2.  **Start the FastAPI Server:**
    Use `uvicorn` to run the main application.
    ```bash
    uvicorn src.myapp.main:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

## Running the Benchmarks

To evaluate the performance of the RAG pipeline, you can run the benchmark scripts directly.

-   **To compare chunking strategies:**
    ```bash
    python -m benchmarks.compare_chunkers
    ```

-   **To compare similarity metrics:**
    ```bash
    python -m benchmarks.compare_similarity
    ```

Results will be printed to the console and saved to a JSON file in the `benchmarks/` directory.
