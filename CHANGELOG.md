# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-12-06

### Added
- **Dynamic Intent Detection**: New `intent.py` module that automatically detects user intent (Tutorial, Conversation, Factual) to adjust system behavior.
- **Neural Reranking**: Integrated `sentence-transformers` CrossEncoder to re-rank RAG results, significantly improving relevance for complex queries.
- **Debug Flag**: Added `--debug` command-line argument to view internal retrieval and scoring logs.

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
