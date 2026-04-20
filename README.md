# MedRAG: A Retrieval-Augmented Medical Question Answering System with Verifiable Citations

MedRAG is an end-to-end medical question answering project built around retrieval-augmented generation (RAG). The system retrieves relevant medical evidence from NIH-aligned data, compares multiple retrieval strategies, and presents grounded outputs with traceable citations.

## Project Goal

Medical QA requires accuracy, transparency, and trust. Large language models can sound fluent while hallucinating unsupported claims. MedRAG reduces that risk by grounding every answer in retrieved source material, with explicit citations traceable back to the original NIH source in the dataset.

The system can:
- Answer medical questions using retrieved evidence rather than unsupported generation
- Compare sparse, dense, and hybrid retrieval methods side by side
- Support answer verifiability through explicit source citations
- Run entirely locally on GCP with no external API dependencies

## Dataset

The project is designed around MedQuAD, a collection of NIH medical question-answer pairs. Each example includes medical content together with source metadata such as document identity.
The retrieval corpus contains over 10,000 chunked passages standardized into a fixed schema shared across all retrieval methods:

- `chunk_id`, `chunk_text`, `document_id`, `source_page_url`

## Retrieval Methods

**Sparse (BM25)** — The sparse baseline uses BM25 over chunked medical text. It provides a keyword-based baseline that is easy to interpret and useful for comparison.

**Dense** - The dense retriever uses a pre-trained sentence embedding model to encode chunks and queries into vector representations. Chunk embeddings are precomputed and indexed with FAISS for efficient similarity search.

**Hybrid** - The hybrid retriever combines both approaches: (1) BM25 first recalls a candidate set and (2) dense embeddings rerank the candidate chunks. This design supports stronger recall than pure dense retrieval on some queries while still benefiting from semantic reranking.

## Evaluation Results

### Retrieval Method Comparison (k=5)

| Method | Recall@5 | Precision@5 |
|--------|----------|-------------|
| BM25 | 0.5871 | 0.1375 |
| Dense | **0.7469** | **0.1816** |
| Hybrid | 0.7177 | 0.1746 |

Dense retrieval achieved the best Recall@5 (0.747). BM25 struggled due to vocabulary mismatch between short questions and long answer chunks. Hybrid underperformed Dense because the BM25 candidate pool (candidate_k=20) can exclude the correct document before dense reranking sees it.

### Chunking Analysis (BM25, k=5)

| Chunk Size | Overlap | Recall@5 |
|-----------|---------|----------|
| 60 | 0 | 0.5803 |
| 100 | 0 | 0.6020 |
| 150 | 0 | 0.6113 |
| **200** | **0** | **0.6188** |

Larger chunks with no overlap performed best. Small chunks fragment medical terms across boundaries, reducing BM25's keyword signal. Overlap consistently hurt performance by increasing index noise.

**Key takeaway:** Retrieval strategy matters more than chunk size. Even the best BM25 chunking config (Recall@5: 0.619) falls well below Dense retrieval's default (Recall@5: 0.747).


## Repository Structure

```
NIH-Medical-Retrieval-Augmented-Generation/
├── medrag_chunking/       # Document chunking pipeline
├── medrag_eval/           # Evaluation scripts
├── medrag_llm/            # Ollama LLM client and pipeline
├── medrag_retrieval/      # FAISS, BM25, and hybrid retrieval
├── medrag_ui/             # Streamlit UI components
├── artifacts/             # Pre-built chunked data
├── scripts/               # Utility and build scripts
├── app.py                 # Main Streamlit entry point
└── requirements.txt
```

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Llama 3 (via Ollama) |
| Embeddings | sentence-transformers |
| Vector Search | FAISS |
| Lexical Search | BM25 |
| Frontend | Streamlit |
| Dataset | MedQuAD |


## Running Without Docker (Local)

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- CUDA 11.8 compatible GPU (recommended)

### Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/NIH-Medical-Retrieval-Augmented-Generation.git
cd NIH-Medical-Retrieval-Augmented-Generation

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch for CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt

# Pull the Llama 3 model
ollama pull llama3

# Run the app
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Then open http://localhost:8501 in your browser.

---

## Running With Docker

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed
- **For GPU:** NVIDIA Container Toolkit (see below)

### GPU Prerequisites (optional but recommended)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### CPU (for local machines)

```bash
docker compose up --build
```

### GPU (recommended, for VM with NVIDIA GPU)

```bash
docker compose -f docker-compose.gpu.yml up --build
```

### Pull the Llama 3 model (first time only)

In a separate terminal after the containers are running:

```bash
docker exec ollama ollama pull llama3
```

Then open http://localhost:8501 in your browser.

> **Note:** The first query may take 2-3 minutes while the dense index is built and the LLM is loaded into GPU memory. Subsequent queries will be significantly faster.

---

## Data Preparation & Chunking

The pre-built `artifacts/chunks/medquad_chunks.csv` is already included in the repo, so this section is only needed if you want to rebuild from scratch.

### Step 1 — Download and preprocess the raw dataset

Download `medquad.csv` from [Kaggle](https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research), then run:

```bash
python3 scripts/preprocess_medquad.py \
  --input medquad.csv \
  --output medquad_preprocessed.csv
```

This cleans the raw data by normalizing text, removing low-quality rows, and deduplicating entries.

### Step 2 — Build chunks

```bash
python3 scripts/build_chunks.py \
  --input medquad_preprocessed.csv \
  --output artifacts/chunks/medquad_chunks.csv \
  --text-column answer \
  --document-id-column document_id \
  --source-url-column source_page_url
```

The chunking pipeline outputs a CSV with the stable schema used by all retrievers (`chunk_id`, `chunk_text`, `document_id`, `source_page_url`). 

Note: `source_page_url` is currently left empty as we did not tie a external data source web url for the scope of this project thus far, but can be added if needed. However it is integral to the schema expected in the project files.

---

## Retrieval Demos (Optional)

After setup, you can test individual retrieval components directly (with example data below, or you can replace with "--chunks artifacts/chunks/medquad_chunks.csv \"):

Run BM25:

```bash
python3 scripts/run_bm25_demo.py \
  --chunks examples/generated_chunks_example.csv \
  --query "What are symptoms of iron deficiency anemia?" \
  --top-k 2
```

Run dense retrieval:

```bash
python3 scripts/run_dense_demo.py \
  --chunks examples/generated_chunks_example.csv \
  --query "What are symptoms of iron deficiency anemia?" \
  --top-k 2
```

Run hybrid retrieval:

```bash
python3 scripts/run_hybrid_demo.py \
  --chunks examples/generated_chunks_example.csv \
  --query "What are symptoms of iron deficiency anemia?" \
  --top-k 2 \
  --candidate-k 3
```
