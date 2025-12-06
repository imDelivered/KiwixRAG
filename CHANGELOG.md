# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [2.0.0] - 2025-12-06

### Major Features
- **Multi-Joint RAG System**: Introduced a 3-stage reasoning pipeline to eliminate hallucinations:
    - **Joint 1 (Entity Extraction)**: Uses `llama3.2:1b` to identify search entities and aliases before retrieval.
    - **Joint 2 (Article Scoring)**: Uses `qwen2.5:0.5b` to grade article relevance (0-10) before indexing.
    - **Joint 3 (Chunk Filtering)**: Uses `llama3.2:1b` to semantically filter retrieved chunks, ensuring only relevant facts reach the final answer.
- **Robust Setup**: Updated `setup.sh` to explicitly handle all dependencies (`requests`, `ollama`) and pull all required joint models automatically.
- **Enhanced Debugging**: `krag --debug` now provides color-coded, real-time tracing of all three joints and the retrieval process.

### Changed
- **Default Models**: Switched default generation model to `llama3.2:1b` for better reasoning and speed.
- **Clean Installation**: Fixed a critical regression where the joint system failed to checking in clean environments due to missing python packages.
- **Launcher**: Updated `run_chatbot.py` to robustly handle accidental query arguments.

### Removed
- Removed purely keyword-based retrieval fallback in favor of the Multi-Joint approach.

## [1.1.0] - 2025-12-06

### Added
- **Dynamic Intent Detection**: New `intent.py` module that automatically detects user intent (Tutorial, Conversation, Factual) to adjust system behavior.
- **Neural Reranking**: Integrated `sentence-transformers` CrossEncoder to re-rank RAG results, significantly improving relevance for complex queries.
- **Debug Flag**: Added `--debug` command-line argument to view internal retrieval and scoring logs.
- **Semantic Title Search**: Implemented vector-based JIT discovery (`build_title_index.py`) allowing the system to find relevant articles even when keywords don't match (e.g., "King of Pop" -> "Michael Jackson").

### Changed
- **Retrieval Depth**: Increased `top_k` documents from 3 to 5 to improve recall for obscure facts.
- **Prompt Engineering**:
    - Fixed specific citation hallucinations (removed misleading "Tupac" example).
    - Hardened refusal instructions for partial context matches.
    - Added dynamic system instructions based on detected intent (e.g., "Step-by-step" for tutorials).
- **Conversation Flow**: Greetings and chit-chat now skip the RAG lookup entirely for faster, more natural responses.

## [0.1.0] - 2025-12-05
### Added
- Initial release of KiwixRAG.
- Offline RAG system using FAISS and LanceDB/LibZIM.
- Basic GUI with Tkinter.
