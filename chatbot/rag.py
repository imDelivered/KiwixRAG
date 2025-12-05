import os
import shutil
import pickle
from typing import List, Dict, Optional, Tuple, Iterator
import libzim
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

class TextProcessor:
    """Handles text extraction and chunking."""
    
    @staticmethod
    def extract_text(html_content: bytes) -> str:
        """Extract clean text from HTML."""
        try:
            if isinstance(html_content, str):
                text_content = html_content
            else:
                text_content = bytes(html_content).decode('utf-8', errors='ignore')
        except Exception:
            text_content = ""
            
        if not text_content:
            return ""

        try:
            soup = BeautifulSoup(text_content, 'html.parser')
            # Remove script and style elements
            for tag in ["script", "style", "header", "footer", "nav"]:
                for element in soup.find_all(tag):
                    element.decompose()
            
            text = soup.get_text(separator=' ', strip=True)
            return text
        except Exception as e:
            # print(f"BS4 Error: {e}")
            return ""

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        if not words:
            return []
            
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)
        return chunks

class RAGSystem:
    """Core RAG System handling indexing and retrieval."""
    
    def __init__(self, index_dir: str = "data/index", model_name: str = "all-MiniLM-L6-v2"):
        self.index_dir = index_dir
        self.model_name = model_name
        self.encoder = None
        self.faiss_index = None
        self.bm25 = None
        self.documents = [] # Metadata storage
        self.doc_chunks = [] # Actual text chunks
        self.indexed_paths = set() # Track what we have indexed
        
        # Paths
        self.faiss_path = os.path.join(index_dir, "faiss.index")
        self.bm25_path = os.path.join(index_dir, "bm25.pkl")
        self.meta_path = os.path.join(index_dir, "metadata.pkl")
        
    def load_resources(self):
        """Load models and indices if they exist."""
        print(f"Loading encoder: {self.model_name}...")
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            self.encoder = SentenceTransformer(self.model_name, device=device)
        except:
             self.encoder = SentenceTransformer(self.model_name)
        
        if os.path.exists(self.faiss_path) and os.path.exists(self.bm25_path):
            print("Loading existing indices...")
            self.faiss_index = faiss.read_index(self.faiss_path)
            with open(self.bm25_path, 'rb') as f:
                self.bm25 = pickle.load(f)
            with open(self.meta_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.doc_chunks = data['chunks']
                # Rebuild indexed set
                self.indexed_paths = {doc.get('path') for doc in self.documents if doc.get('path') is not None}
        else:
            print("No existing indices found. Need to build index.")

    def build_index(self, zim_path: str, limit: int = None, batch_size: int = 1000):
        """Build FAISS and BM25 indices from ZIM file using batch processing."""
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        if not self.encoder:
            self.encoder = SentenceTransformer(self.model_name, device=device)
            
        os.makedirs(self.index_dir, exist_ok=True)
        
        print(f"Opening ZIM file: {zim_path}")
        zim = libzim.Archive(zim_path)
        
        # Initialize indices
        self.doc_chunks = []
        self.documents = []
        
        # Approximate dimension from model
        sample_emb = self.encoder.encode(["test"], device=device)
        dimension = sample_emb.shape[1]
        
        self.faiss_index = faiss.IndexFlatL2(dimension)
        
        count = 0
        total = zim.entry_count
        
        current_batch_chunks = []
        current_batch_meta = []
        
        print("Extracting and indexing documents in batches...")
        for i in tqdm(range(total)):
            try:
                entry = zim._get_entry_by_id(i)
                if entry.is_redirect:
                    continue
                    
                item = entry.get_item()
                if item.mimetype != 'text/html':
                    continue
                    
                text = TextProcessor.extract_text(item.content)
                if not text:
                    continue
                    
                doc_chunks = TextProcessor.chunk_text(text)
                for chunk in doc_chunks:
                    current_batch_chunks.append(chunk)
                    current_batch_meta.append({
                        'title': entry.title,
                        'path': entry.path,
                        'zim_index': i
                    })
                    
                count += 1
                
                # Process batch
                if len(current_batch_chunks) >= batch_size:
                    self._process_batch(current_batch_chunks, current_batch_meta)
                    current_batch_chunks = []
                    current_batch_meta = []
                    
                if limit and count >= limit:
                    break
            except Exception as e:
                # print(f"Error processing entry {i}: {e}")
                pass

        # Process remaining
        if current_batch_chunks:
            self._process_batch(current_batch_chunks, current_batch_meta)

        # Save indices
        faiss.write_index(self.faiss_index, self.faiss_path)
        
        print("Building BM25 index...")
        tokenized_corpus = [chunk.split(" ") for chunk in self.doc_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        with open(self.bm25_path, 'wb') as f:
            pickle.dump(self.bm25, f)
            
        with open(self.meta_path, 'wb') as f:
            pickle.dump({'documents': self.documents, 'chunks': self.doc_chunks}, f)
            
        print("Indexing complete.")

    def _process_batch(self, chunks: List[str], meta: List[Dict]):
        """Encode and index a batch of chunks."""
        if not chunks:
            return
            
        # Add to storage
        self.doc_chunks.extend(chunks)
        self.documents.extend(meta)
        
        # Encode
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = self.encoder.encode(chunks, batch_size=32, device=device, show_progress_bar=False, convert_to_numpy=True)
        
        # Add to FAISS
        self.faiss_index.add(embeddings.astype('float32'))

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Hybrid retrieval with Just-In-Time indexing."""
        if not self.faiss_index or not self.bm25:
             pass # Logic handled in init/load
        
        print(f"\n🔍 Processing Query: '{query}'")
        
        # 0. JIT Indexing step
        try:
            title_candidates = self.search_by_title(query, full_text=True)
            if title_candidates:
                candidate_titles = [c['metadata']['title'] for c in title_candidates]
                print(f"🔎 Found Title Candidates: {candidate_titles}")
            else:
                print("🔎 No direct title matches found.")
            
            for cand in title_candidates:
                path = cand['metadata'].get('path')
                # Check if already indexed
                if path is not None and path not in self.indexed_paths:
                    print(f"⚙️  JIT Indexing: '{cand['metadata']['title']}' (New Topic)...")
                    text = cand['text'] # Full text
                    chunks = TextProcessor.chunk_text(text)
                    
                    if chunks:
                        import torch
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        if not self.encoder:
                            self.encoder = SentenceTransformer(self.model_name, device=device)
                            
                        embeddings = self.encoder.encode(chunks, device=device, show_progress_bar=False)
                        
                        if self.faiss_index is None:
                            dimension = embeddings.shape[1]
                            self.faiss_index = faiss.IndexFlatL2(dimension)
                            
                        self.faiss_index.add(embeddings.astype('float32'))
                        
                        self.doc_chunks.extend(chunks)
                        for _ in chunks:
                            self.documents.append(cand['metadata'])
                            
                        self.indexed_paths.add(path)
                        print(f"✅ Indexed {len(chunks)} chunks.")
        except Exception as e:
            print(f"❌ JIT Error: {e}")

        # 1. Dense Retrieval
        print("🧠 Performing Dense Retrieval...")
        dense_hits = {}
        if self.faiss_index and self.faiss_index.ntotal > 0:
            try:
                q_emb = self.encoder.encode([query]).astype('float32')
                k_search = min(top_k * 2, self.faiss_index.ntotal)
                D, I = self.faiss_index.search(q_emb, k_search)
                dense_hits = {idx: rank for rank, idx in enumerate(I[0])}
                print(f"✅ Dense retrieval found {len(dense_hits)} hits.")
            except Exception as e: 
                print(f"❌ Dense search error: {e}")
                pass
        else:
            print("⚠️ FAISS index not available or empty for dense retrieval.")
        
        # 2. Sparse Retrieval
        print("📝 Performing Sparse Retrieval (BM25)...")
        sparse_hits = {}
        if self.bm25:
            try:
                tokenized_query = query.split(" ")
                sparse_scores = self.bm25.get_scores(tokenized_query)
                sparse_indices = np.argsort(sparse_scores)[::-1][:top_k * 2]
                sparse_hits = {idx: rank for rank, idx in enumerate(sparse_indices)}
                print(f"✅ Sparse retrieval found {len(sparse_hits)} hits.")
            except Exception as e: 
                print(f"❌ Sparse search error: {e}")
                pass
        else:
            print("⚠️ BM25 index not available for sparse retrieval.")
            
        # 3. Reciprocal Rank Fusion
        print("융 Combining results with RRF...")
        fused_scores = {}
        all_indices = set(dense_hits.keys()) | set(sparse_hits.keys())
        k = 60
        
        for idx in all_indices:
            if idx == -1: continue 
            score = 0
            if idx in dense_hits:
                score += 1 / (k + dense_hits[idx])
            if idx in sparse_hits:
                score += 1 / (k + sparse_hits[idx])
            fused_scores[idx] = score
            
        sorted_hits = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        print("\n📄 Semantic Search Results:")
        for idx, score in sorted_hits:
            if idx < len(self.doc_chunks):
                doc = self.documents[idx]
                print(f"   - {doc['title']} (Score: {score:.4f})")
                results.append({
                    'text': self.doc_chunks[idx],
                    'metadata': doc,
                    'score': score
                })
        
        return results

    def search_by_title(self, query: str, zim_path: str = None, full_text: bool = False) -> List[Dict]:
        """Fast fallback: Search by title using ZIM's internal index."""
        if not zim_path:
            # Try to find one in current dir
            files = [f for f in os.listdir('.') if f.endswith('.zim')]
            if files:
                zim_path = files[0]
            else:
                return []
                
        try:
            zim = libzim.Archive(zim_path)
            searcher = libzim.SuggestionSearcher(zim)
            
            clean_query = query.replace("?", "").replace(".", "").replace("!", "")
            tokens = clean_query.split()
            
            # Expanded stopword list
            stopwords = {
                "how", "what", "when", "who", "where", "why", "is", "are", "do", "did", "does", "can", "could",
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "about",
                "tell", "me", "explain", "describe", "define", "show", "list", "give"
            }
            
            # Filter tokens (case-insensitive check)
            keywords = [w for w in tokens if w.lower() not in stopwords]
            
            hits_map = {} # path -> hit
            
            # Strategy 1: Try full combined phrase (e.g. "European Union")
            if keywords:
                full_term = " ".join(keywords)
                # print(f"🔎 Strategy 1: '{full_term}'")
                results = searcher.suggest(full_term)
                if results.getEstimatedMatches() > 0:
                     self._collect_hits(zim, results, hits_map, full_text)
            
            # Strategy 2: If no or few results, try individual keywords (longest first)
            if len(hits_map) < 3 and keywords:
                # Sort keywords by length
                sorted_keywords = sorted(keywords, key=len, reverse=True)
                for kw in sorted_keywords[:2]: # Try top 2 longest keywords
                    # print(f"🔎 Strategy 2: '{kw}'")
                    results = searcher.suggest(kw)
                    if results.getEstimatedMatches() > 0:
                         self._collect_hits(zim, results, hits_map, full_text)
                         
            # Strategy 3: Fallback to original query if nothing (last resort)
            if not hits_map:
                 results = searcher.suggest(query)
                 if results.getEstimatedMatches() > 0:
                     self._collect_hits(zim, results, hits_map, full_text)

            return list(hits_map.values())[:5] # Return top 5 unique

        except Exception as e:
            # print(f"Fast Search Error: {e}")
            return []

    def _collect_hits(self, zim, results, hits_map: Dict, full_text: bool):
        """Helper to collect and process hits."""
        s_hits = results.getResults(0, 5) # Top 5 per strategy
        for hit_path in s_hits:
            if hit_path in hits_map:
                continue
            try:
                try:
                    entry = zim.get_entry_by_path(hit_path)
                except:
                    entry = zim.get_entry_by_title(hit_path)
                    
                item = entry.get_item()
                if item.mimetype == 'text/html':
                    text = TextProcessor.extract_text(item.content)
                    display_text = text if full_text else text[:2000]
                    
                    hits_map[hit_path] = {
                        'text': display_text, 
                        'metadata': {'title': entry.title, 'path': entry.path},
                        'score': 1.0
                    }
            except:
                continue

if __name__ == "__main__":
    pass
