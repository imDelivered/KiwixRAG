# KiwixRAG

A powerful offline-capable chatbot with **Retrieval-Augmented Generation (RAG)** that lets you chat with AI using local knowledge bases like Wikipedia, Python documentation, or any ZIM file archive.

> **⚠️ Platform Note:** This software is currently only available for Linux. Windows and macOS support may be added in the future.

---


---

## Disclaimer

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DISCLAIMER OF LIABILITY                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ This software is provided "as is" without warranty of any kind. The        │
│ author(s) and contributors are not responsible for any misuse, damage, or   │
│ consequences arising from the use of this software.                         │
│                                                                             │
│ Users are solely responsible for:                                           │
│ • Compliance with applicable laws and regulations                           │
│ • Ethical use of AI technology                                              │
│ • Content generated or accessed through this software                        │
│ • Any actions taken based on information from this software                 │
│ • Verifying the accuracy of AI-generated content                            │
│                                                                             │
│ By using this software, you agree to use it responsibly and accept full     │
│ liability for your actions.                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Features

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ • AI Chat Interface: Beautiful GUI built with tkinter                      │
│ • Offline Knowledge Base: Works with ZIM files                             │
│ • Hybrid Search: Semantic (FAISS) + keyword (BM25) search                  │
│ • Just-In-Time Indexing: Auto-indexes articles on-the-fly                  │
│ • Modern UI: Dark/light mode, autocomplete, shortcuts                      │
│ • Multiple Models: Switch between any Ollama model                          │
│ • Multi-Joint RAG: Three reasoning models prevent hallucinations           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## How It Works

### Multi-Joint RAG Architecture

KiwixRAG uses an advanced **multi-joint architecture** where small AI reasoning models work together to ensure accurate, hallucination-free responses.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MULTI-JOINT RAG PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User Query                                                                 │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  JOINT 1: Entity Extraction                                        │   │
│  │  → Identifies core entities and resolves aliases for search        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Dual-Path Search                                                  │   │
│  │  → Parallel Semantic (Vector) and Keyword (BM25) discovery         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  JOINT 2: Article Scoring                                          │   │
│  │  → Evaluates candidate articles for relevance to the entity        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Just-In-Time Indexing                                             │   │
│  │  → Dynamically chunks and indexes only high-relevance content      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Hybrid Retrieval & Fusion                                         │   │
│  │  → Retrieves chunks and ranks them via Reciprocal Rank Fusion      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  JOINT 3: Chunk Filtering                                          │   │
│  │  → Semantic evaluation of chunks against the original query        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Final LLM Generation                                              │   │
│  │  → Synthesizes answer exclusively from verified context            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

![Architecture Diagram](architecture_diagram.png)

### Key Innovation: Reasoning Joints

Traditional RAG systems often retrieve irrelevant information, leading to hallucinations. KiwixRAG solves this with three specialized reasoning models:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  JOINT 1: Entity Extraction                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Understands what you're really asking about                            │
│  • Extracts entities and discovers aliases automatically                   │
│  • Prevents searching for wrong topics                                     │
│  • Model: llama3.2:1b (~500ms)                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  JOINT 2: Article Scoring                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Evaluates Wikipedia article relevance (0-10 scale)                      │
│  • Selects only the most relevant articles for indexing                    │
│  • Ensures high-quality knowledge base                                    │
│  • Model: qwen2.5:0.5b (~400ms)                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  JOINT 3: Chunk Filtering                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Filters retrieved text chunks by query relevance                        │
│  • Removes off-topic information                                           │
│  • Keeps only content that directly answers your question                 │
│  • Model: llama3.2:1b (~400ms)                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### System Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Hybrid Retrieval Pipeline                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Dense Search (FAISS): Semantic similarity using embeddings             │
│  • Sparse Search (BM25): Keyword-based matching                           │
│  • Reciprocal Rank Fusion: Combines both methods for best results         │
└─────────────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Just-In-Time Indexing                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  • No need to pre-index entire ZIM files                                  │
│  • Articles indexed on-the-fly as needed                                  │
│  • Efficient memory usage                                                  │
│  • Faster startup, slower first queries                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Modern GUI                                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Real-time streaming responses                                           │
│  • Query history and autocomplete                                          │
│  • Keyboard shortcuts and quick queries                                    │
│  • Dark/light mode toggle                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Setup

### Prerequisites

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ • Linux (tested on Ubuntu/Debian)                                           │
│ • Python 3.8+                                                               │
│ • Internet connection (for initial setup)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Installation Steps

**Step 1: Run the setup script**
```bash
chmod +x setup.sh
./setup.sh
```

> **What the setup script does:**
> - Installs Python and dependencies
> - Sets up a virtual environment
> - Installs Ollama
> - Downloads AI models:
>   - llama3.2:1b (default response model + Joint 1 & 3)
>   - llama3.1:1b (alternative response model, optional)
>   - qwen2.5:0.5b (Joint 2: article scoring)
> - Enables Ollama service
> - Creates a `krag` command for easy access
**Step 2: Add a ZIM file (optional but recommended)**
- Download a ZIM file (e.g., from [Kiwix](https://library.kiwix.org/))
- Place it in the project directory (e.g., `wikipedia_en_all_maxi_2025-08.zim`)
- The chatbot will automatically detect and use it

**Step 3: Start the chatbot**
```bash
krag
```

Or run manually:
```bash
./run_chatbot.sh
```
<img width="895" height="700" alt="Screenshot_20251205_173826" src="https://github.com/user-attachments/assets/f4509a75-dd30-4344-af13-f1a5d5c293b6" />


---

## Manual Setup (Alternative)

> **If you prefer manual setup instead of the automated script:**

```bash
# Install system dependencies
sudo apt install python3 python3-venv python3-tk

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.2:1b

# Run the chatbot
python3 run_chatbot.py
```

---

## Usage

### Basic Commands

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Command          │ Action                                                    │
├──────────────────┼───────────────────────────────────────────────────────────┤
│ /help            │ Show help menu                                            │
│ /clear           │ Clear chat history                                        │
│ /dark            │ Toggle dark/light mode                                    │
│ /model           │ Switch to a different Ollama model                        │
│ /exit or :q      │ Quit the application                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Keyboard Shortcuts

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Shortcut              │ Action                                               │
├───────────────────────┼──────────────────────────────────────────────────────┤
│ Enter                 │ Send message                                         │
│ Highlight + Enter     │ Auto-paste and query selected text                  │
│ Ctrl+Click            │ Select word and query it                             │
│ ↑↓                    │ Navigate autocomplete suggestions                    │
│ Tab                   │ Select autocomplete suggestion                       │
│ Esc                   │ Close dialogs                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Building a Full Index (Optional)

For faster retrieval on large ZIM files, you can pre-build an index:

```bash
python3 build_index.py --zim wikipedia_en_all_maxi_2025-08.zim
```

> **Note:** This creates a full FAISS + BM25 index in `data/index/`. Without this, the system uses Just-In-Time indexing (works great, just slightly slower on first queries).

---

## Configuration

### Default Model

Edit `chatbot/config.py` to change the default model:

```python
DEFAULT_MODEL = "llama3.2:1b"  # Change to your preferred model
```

### RAG Settings

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Setting              │ Value                                                 │
├──────────────────────┼───────────────────────────────────────────────────────┤
│ Embedding Model      │ all-MiniLM-L6-v2 (fast, efficient)                    │
│ Top-K Results        │ 5 chunks per query (configurable)                    │
│ Chunk Size           │ 500 words with 50-word overlap                       │
│ Joint System         │ Enabled by default (can disable in config.py)         │
│ Strict RAG Mode      │ Enabled (requires context to answer)                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

> Configurable in `chatbot/rag.py` and `chatbot/config.py`

---

## Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Package                │ Purpose                                            │
├────────────────────────┼────────────────────────────────────────────────────┤
│ libzim                 │ ZIM file reading                                   │
│ sentence-transformers  │ Text embeddings                                    │
│ faiss-cpu              │ Vector similarity search                           │
│ rank_bm25              │ Keyword search                                     │
│ beautifulsoup4         │ HTML parsing                                       │
│ numpy                  │ Numerical operations                               │
│ tqdm                   │ Progress bars                                      │
│ requests               │ HTTP requests                                     │
│ ollama                 │ Ollama API client                                  │
│ tkinter                │ GUI (usually pre-installed with Python)             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> All dependencies are installed automatically by `setup.sh`.

---

## Use Cases

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ • Offline Wikipedia    │ Ask questions without internet                     │
│ • Documentation Chat   │ Chat with Python docs, manuals                     │
│ • Research Assistant   │ Query large knowledge bases locally                │
│ • Educational Tool     │ Learn from offline encyclopedias                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Note:** While this system can help explore knowledge bases, remember that AI responses may contain errors or fabricated information. Always cross-reference important facts with authoritative sources.

---

## Troubleshooting

### "tkinter not available"
```bash
sudo apt install python3-tk
```

### "Cannot reach Ollama"
Make sure Ollama is running:
```bash
ollama serve
```

### "No models found"
Install required models:
```bash
ollama pull llama3.2:1b  # Default model + joints
ollama pull qwen2.5:0.5b  # Article scoring joint
```

### Slow first queries
> This is normal with Just-In-Time indexing. The system indexes articles as needed. For faster performance, build a full index (see above).

---

## Notes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ • Works without ZIM file, but no RAG capabilities                           │
│ • First-time setup downloads ~1-2GB (Ollama + models + deps)                │
│ • GPU acceleration automatic if CUDA available                              │
│ • All data stays local - no internet required after setup                  │
│ • Joint system adds ~1.3s latency but significantly improves accuracy      │
│ • Can disable joints in config.py for faster (but less accurate) responses │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Resources

- [Ollama Models](https://ollama.ai/library)
- [Kiwix ZIM Files](https://library.kiwix.org/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

---

## License

See repository for license information.

---

**Enjoy chatting with your offline AI assistant!**
