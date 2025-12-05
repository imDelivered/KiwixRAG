#!/usr/bin/env python3
import argparse
import os
import sys

# Ensure the current directory is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from chatbot.rag import RAGSystem
except ImportError:
    # Try assuming we are inside the package or something went wrong with path
    sys.path.append(os.getcwd())
    from chatbot.rag import RAGSystem

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Search Index for KiwixRAG")
    parser.add_argument("--zim", required=True, help="Path to the .zim file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of articles to index (for testing)")
    args = parser.parse_args()

    if not os.path.exists(args.zim):
        print(f"Error: File not found: {args.zim}")
        sys.exit(1)

    print(f"Initializing Indexer for: {args.zim}")
    dag = RAGSystem()
    dag.build_index(args.zim, limit=args.limit)
    print("Optimization Complete.")
