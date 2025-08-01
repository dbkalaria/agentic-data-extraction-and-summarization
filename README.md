# Intelligent Data Extraction & Summarization with Agentic Workflows on GCP

This project extracts key information and generate summaries from unstructured text documents using GCP services. This information retrieval and summaries are then used by the agent as an tools to answer the user's query.

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/agentic-data-extraction-and-summarization.git
    cd agentic-data-extraction-and-summarization
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download spaCy model:**
    After installing the dependencies, download the required spaCy English model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

4.  **GCP Authentication:**
    Ensure you have authenticated with Google Cloud. You can do this by setting up Application Default Credentials:
    ```bash
    gcloud auth application-default login
    ```
    Alternatively, you can set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of your service account key file.

5.  **Environment Variables:**
    Create a `.env` file in the root directory based on `.env.sample` and populate it with necessary environment variables (e.g., GCP project ID, bucket names).

## Dataset Source

This project uses XSUM dataset which is an unstructured news dataset with extreme summary (single sentence) from the hugging face.

## GCP Resource Requirements

The following Google Cloud Platform services are utilized or recommended for this project:

*   **Google Cloud Storage:** For storing raw and processed text documents.
*   **Vertex AI:** For leveraging Generative AI models (e.g., Gemini) for summarization and potentially custom models for information extraction.
*   **Google Cloud Natural Language API:** For entity extraction, sentiment analysis, and other NLP tasks.
*   **Vertex AI Vector Search (formerly Matching Engine):** For efficient similarity search on embeddings, useful for RAG (Retrieval Augmented Generation) or semantic search.
*   **Google Cloud Firestore:** For potential memory management and storing metadata in agentic workflows.

Ensure that your GCP project has the necessary APIs enabled and that the service account used has appropriate permissions to access these services.

## GCP Service Setup

Follow these steps to set up the required GCP services in your project:

1.  **Enable APIs:**
    Navigate to the Google Cloud Console and enable the following APIs for your project:
    *   **Vertex AI API**
    *   **Cloud Natural Language API**
    *   **Cloud Storage API**
    *   **Firestore API** 

    You can enable them via the `gcloud` CLI as well:
    ```bash
    gcloud services enable aiplatform.googleapis.com
    gcloud services enable language.googleapis.com
    gcloud services enable storage.googleapis.com
    gcloud services enable firestore.googleapis.com 
    ```

2.  **Google Cloud Storage:**
    *   Create a new Cloud Storage bucket to store your raw and processed text documents.
    *   Ensure the service account used by your application has `Storage Object Admin` or `Storage Object Viewer` and `Storage Object Creator` roles on this bucket.
    ```bash
    gsutil mb gs://your-unique-bucket-name
    ```

3.  **Vertex AI:**
    *   No specific setup is required beyond enabling the Vertex AI API for using pre-trained models like Gemini.

4.  **Google Cloud Natural Language API:**
    *   No specific setup is required beyond enabling the API.

5.  **Vertex AI Vector Search:**
    *   **Enable the API:** Ensure `aiplatform.googleapis.com` is enabled (as above).
    *   **Create an Index:** You will need to create a Vector Search index to store your embeddings. This typically involves:
        *   Defining the index configuration (e.g., number of dimensions, distance measure).
    *   **Deploy an Index Endpoint:** Once the index is built, you need to deploy it to an endpoint for querying.
    *   Refer to the official Vertex AI Vector Search documentation for detailed steps on creating and deploying indexes.

6.  **Google Cloud Firestore:**
    *   **Choose a mode:** When you first enable Firestore, you'll be prompted to choose between Native mode and Datastore mode. For most new applications, Native mode is recommended.
    *   **Select a location:** Choose a regional or multi-regional location for your Firestore database.

Remember to replace `your-unique-bucket-name` with an actual unique bucket name for your project.

## Core Scripts

This project consists of several key Python scripts that orchestrate the data pipeline and agentic workflow:

### `data_ingestion.py`

This script is responsible for downloading the raw dataset from its source and uploading it to a specified Google Cloud Storage bucket. It acts as the initial step in the data pipeline, ensuring that the data is available in GCP for further processing.

**How to Execute:**
```bash
python src/data_ingestion.py
```
Before running, ensure your `.env` file is configured with the necessary source URLs and GCS bucket details.

### `eda.py`

The `eda.py` script performs basic Exploratory Data Analysis (EDA) on the ingested dataset. It downloads a data split from Google Cloud Storage, loads it into a Pandas DataFrame, and calculates statistics such as document and summary lengths. This script is useful for understanding the characteristics of your data.

**How to Execute:**
```bash
python src/eda.py
```
Ensure your `.env` file is configured with the `GCS_BUCKET_NAME` and that data has been ingested using `data_ingestion.py`.

### `main.py`

The `main.py` script is the central orchestrator for processing the ingested data. It performs the following key operations:

1.  **Data Loading:** Loads the raw text documents from the Google Cloud Storage bucket.
2.  **Information Extraction:** Utilizes the Google Cloud Natural Language API or custom models (if implemented) to extract key entities, facts, and other relevant information from the documents.
3.  **Summary Generation:** Generates concise summaries of each document using Vertex AI's Generative AI models (e.g., Gemini).
4.  **Embedding Generation & Vector Search Upload:** Creates embeddings for the processed documents and uploads them to a Vertex AI Vector Search index for efficient similarity search.
5.  **Document Storage in Firestore:** Stores the processed documents, extracted information, and summaries in Google Cloud Firestore, providing a structured and queryable repository.

**How to Execute:**
```bash
python main.py
```
Ensure all GCP services are set up and your `.env` file contains the correct project ID, bucket names, Vector Search index details, and Firestore collection names.

### `agent.py`

The `agent.py` script demonstrates the agentic workflow, where an AI agent leverages the capabilities provided by the `main.py` processing pipeline as modular "tools." This script showcases how an intelligent agent can answer complex user queries or perform multi-step tasks by orchestrating calls to the information extraction, summarization, and vector search functionalities.

The agent's working mechanism typically involves:
*   **Goal Definition:** The agent is given a specific task or query.
*   **Tool Orchestration:** It intelligently decides which "tools" (e.g., search Vector Search for relevant documents, extract information from a document, summarize a document) to use to achieve its goal.
*   **Reasoning & Planning:** The agent maintains a state, plans its next actions based on the current state and available tools, and handles ambiguity or missing information.

**How to Execute:**
```bash
python src/agent.py
```
The execution of this script will typically involve an interactive prompt or predefined queries to demonstrate the agent's capabilities. Ensure the `main.py` script has been run at least once to populate Vector Search and Firestore with data.

### `evaluate.py`

The `evaluate.py` script is designed for offline evaluation of the summarization and information extraction models. It fetches pre-processed data directly from Firestore, calculates ROUGE scores for the generated summaries against reference summaries, and outputs all collected data and metrics to a CSV file for detailed analysis. This allows for quantitative assessment of the model's performance.

**How to Execute:**
```bash
python src/evaluate.py --limit 100 --output evaluation_results.csv
```
*   `--limit`: Optional. Maximum number of documents to fetch from Firestore (default: all).
*   `--output`: The path to the output CSV file (default: `evaluation_results.csv`).

NOTE: Before running, ensure your `.env` file has `GCP_PROJECT_ID` and `FIRESTORE_COLLECTION` set, and that `main.py` has been executed to populate Firestore with data.