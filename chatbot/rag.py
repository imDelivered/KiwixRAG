
# Hermit - Offline AI Chatbot for Wikipedia & ZIM Files
# Copyright (C) 2026 Hermit-AI, Inc.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
RAG System Implementation.
Handles indexing, retrieval, and context generation for the chatbot.
"""

import os
import sys
import pickle
import numpy as np
import time
from typing import List, Dict, Optional, Tuple, Set

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Warning: RAG dependencies missing (faiss, sentence-transformers, rank_bm25). RAG will error.")
    faiss = None
    SentenceTransformer = None
    BM25Okapi = None

import libzim

from chatbot import config
from chatbot.debug_utils import debug_print
from chatbot.text_processing import TextProcessor

class RAGSystem:
    def __init__(self, index_dir: str = "data/indices", load_existing: bool = True):
        self.index_dir = index_dir
        self.encoder = None
        self.model_name = 'all-MiniLM-L6-v2'
        
        # In-memory storage
        self.faiss_index = None # JIT Index (Vectors)
        self.documents = []     # Metadata
        self.doc_chunks = []    # Text Chunks
        
        # State Tracking
        self.indexed_paths: Set[str] = set()
        self._next_doc_id = 0
        self._chunk_id = 0     # Global chunk ID counter
        
        self.bm25 = None
        self.tokenized_corpus = [] # For BM25
        
        # Title Indices (Pre-computed)
        self.title_faiss_index = None
        self.title_metadata = None
        
        # Paths
        os.makedirs(index_dir, exist_ok=True)
        self.faiss_path = os.path.join(index_dir, "content_index.faiss")
        self.meta_path = os.path.join(index_dir, "content_meta.pkl")
        self.bm25_path = os.path.join(index_dir, "content_bm25.pkl")
        
        self.title_faiss_path = os.path.join(index_dir, "title_index.faiss")
        self.title_meta_path = os.path.join(index_dir, "title_meta.pkl")

        # Initialize SentenceTransformer early (lazy load usually, but we need it for everything)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.encoder = SentenceTransformer(self.model_name, device=device)
        except Exception as e:
            print(f"Failed to load embedding model: {e}")

        # Multi-Joint System Configuration
        self.use_joints = config.USE_JOINTS
        self.entity_joint = None
        self.scorer_joint = None
        self.filter_joint = None
        self.fact_joint = None
        
        # Load Existing Content Indices
        if load_existing and os.path.exists(self.faiss_path) and os.path.exists(self.meta_path):
            print("Loading Content Indices...")
            self.faiss_index = faiss.read_index(self.faiss_path)
            if os.path.exists(self.bm25_path):
                with open(self.bm25_path, 'rb') as f:
                    self.bm25 = pickle.load(f)
            with open(self.meta_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.doc_chunks = data['chunks']
                # Rebuild indexed set
                self.indexed_paths = {doc.get('path') for doc in self.documents if doc.get('path') is not None}
        else:
            print("No existing content indices found.")

        # Load Title Indices (Optional)
        if os.path.exists(self.title_faiss_path) and os.path.exists(self.title_meta_path):
            print("Loading Semantic Title Index...")
            try:
                self.title_faiss_index = faiss.read_index(self.title_faiss_path)
                with open(self.title_meta_path, 'rb') as f:
                    self.title_metadata = pickle.load(f)
                print(f"Loaded {len(self.title_metadata)} titles.")
            except Exception as e:
                print(f"Failed to load title index: {e}")
        
        # Initialize Joint System (if enabled)
        if self.use_joints:
            debug_print("Initializing multi-joint RAG system...")
            try:
                from chatbot.joints import EntityExtractorJoint, ArticleScorerJoint, CoverageVerifierJoint, ChunkFilterJoint, FactRefinementJoint, ComparisonJoint, MultiHopResolverJoint
                self.entity_joint = EntityExtractorJoint()
                self.resolver_joint = MultiHopResolverJoint(model=config.MULTI_HOP_JOINT_MODEL)  # Joint 0.5: Multi-Hop Resolver (Smart 7B)
                self.scorer_joint = ArticleScorerJoint()
                self.coverage_joint = CoverageVerifierJoint()  # Joint 2.5: Coverage Verification
                self.comparison_joint = ComparisonJoint(model=config.COMPARISON_JOINT_MODEL)    # Joint 3.5: Comparison Synthesis (Smart 7B)
                self.filter_joint = ChunkFilterJoint()
                self.fact_joint = FactRefinementJoint()
                debug_print("Joint system initialized successfully (including Multi-Hop Resolver)")
            except Exception as e:
                debug_print(f"Failed to initialize joints: {e}")
                debug_print("Falling back to semantic search")
                self.use_joints = False

    def build_index(self, zim_path: str, limit: int = None, batch_size: int = 1000):
        """
        Build Semantic Title Index from ZIM file.
        This is critical for 'search_by_title' to work effectively.
        """
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        if not self.encoder:
            self.encoder = SentenceTransformer(self.model_name, device=device)
            
        print(f"Scanning ZIM file: {zim_path}")
        zim = libzim.Archive(zim_path)
        
        total_entries = zim.entry_count
        print(f"Total entries: {total_entries}")
        
        if not limit:
            limit = total_entries
            
        # Initialize Title FAISS
        self.title_faiss_index = faiss.IndexFlatIP(384)
        self.title_metadata = []
        
        titles = []
        paths = []
        
        # Iterate ZIM entries (simplified for titles)
        count = 0
        for i in range(total_entries):
            if count >= limit:
                break
                
            entry = zim.get_entry_by_index(i)
            # Filter for articles in namespace 'A'
            if entry.path.startswith("A/"):
                titles.append(entry.title)
                paths.append(entry.path)
                
                # Batch processing
                if len(titles) >= batch_size:
                    embeddings = self.encoder.encode(titles)
                    faiss.normalize_L2(embeddings)
                    self.title_faiss_index.add(embeddings)
                    
                    # Store meta
                    for j, title in enumerate(titles):
                         self.title_metadata.append({
                             'title': title,
                             'path': paths[j]
                         })
                         
                    titles = []
                    paths = []
                    print(f"Indexed {len(self.title_metadata)} titles...")
                    
                count += 1
        
        # Final batch
        if titles:
            embeddings = self.encoder.encode(titles)
            faiss.normalize_L2(embeddings)
            self.title_faiss_index.add(embeddings)
            for j, title in enumerate(titles):
                 self.title_metadata.append({
                     'title': title,
                     'path': paths[j]
                 })
                 
        print(f"Index build complete. Total: {len(self.title_metadata)}")
        
        # Save indices
        print("Saving indices...")
        faiss.write_index(self.title_faiss_index, self.title_faiss_path)
        with open(self.title_meta_path, 'wb') as f:
            pickle.dump(self.title_metadata, f)
        print("Done.")

    def retrieve(self, query: str, top_k: int = 5, mode: str = "FACTUAL", rebound_depth: int = 0, extra_terms: List[str] = None) -> List[Dict]:
        """
        Main Retrieval Function (JIT RAG Pipeline).
        mode: "FACTUAL" (default) or "CODE" (prioritizes code blocks)
        extra_terms: Optional list of improved search terms from a previous failed pass (Adaptive RAG)
        """
        # === EPHEMERAL INDEX RESET ===
        # Reset JIT-specific state to ensure query isolation.
        if rebound_depth == 0:
            debug_print("EPHEMERAL RESET: Clearing JIT index state...")
            self.faiss_index = None  # Will be re-created if articles are found
            self.doc_chunks = []     # Fresh chunks for this query only
            self.documents = []      # Fresh metadata for this query only
            self.indexed_paths = set()  # Reset path tracking
            self._chunk_id = 0       # Reset chunk ID counter for metadata alignment
            debug_print("EPHEMERAL RESET COMPLETE: faiss_index=None, doc_chunks=[], documents=[], indexed_paths={}")
        else:
            debug_print(f"EPHEMERAL RESET SKIPPED (Rebound Depth: {rebound_depth}) - Maintaining index state.")

        # Ensure encoder is loaded
        if not self.encoder:
             import torch
             device = "cuda" if torch.cuda.is_available() else "cpu"
             self.encoder = SentenceTransformer(self.model_name, device=device)

        print(f"\\nProcessing Query: '{query}'")
        
        # 0. JIT Indexing step
        debug_print("-" * 70)
        debug_print("PHASE 0: JUST-IN-TIME INDEXING")
        try:
            # 0a. Entity Extraction (Joint 1)
            entity_info = None
            search_terms = [query]
            if extra_terms:
                search_terms.extend(extra_terms)
                
            is_comparison = False
            
            if self.use_joints and self.entity_joint:
                debug_print("0a. Entity extraction with Joint 1...")
                entity_info = self.entity_joint.extract(query)
                
                # Joint 0.5: Multi-Hop Resolution
                if self.resolver_joint:
                    try:
                        indirects = self.resolver_joint.detect_indirect_references(entity_info)
                        for ind in indirects:
                            target = ind['target']
                            debug_print(f"Resolving multi-hop: '{ind['indirect_entity']}' via '{target}'...")
                            
                            # Target search (High Priority)
                            target_candidates = self.search_by_title(target, full_text=True)
                            if target_candidates:
                                 target_text = target_candidates[0]['text']
                                 resolved_name = self.resolver_joint.resolve_indirect_reference(ind, target_text)
                                 if resolved_name:
                                     new_entity = {
                                         'name': resolved_name,
                                         'type': 'person',
                                         'aliases': []
                                     }
                                     if 'entities' in entity_info:
                                        entity_info['entities'].append(new_entity)
                                     search_terms.append(resolved_name)
                                     print(f"Resolved '{ind['indirect_entity']}' -> {resolved_name}")
                    except Exception as e:
                        debug_print(f"Multi-hop resolution failed: {e}")
                
                # Update search terms
                extracted_names = [e.get('name') for e in entity_info.get('entities', []) if e.get('name')]
                if extracted_names:
                     search_terms = extracted_names + search_terms
                
                is_comparison = entity_info.get('is_comparison', False)
                debug_print(f"Extracted Entities: {extracted_names}")
                debug_print(f"Is Comparison: {is_comparison}")
            
            # 0b. Candidate Selection
            candidates_to_index = []
            seen_titles = set()
            
            debug_print(f"Searching ZIM for terms: {search_terms}")
            
            raw_candidates = []
            for term in search_terms:
                term_results = self.search_by_title(term, full_text=True)
                raw_candidates.extend(term_results)
            
            unique_candidates = []
            for cand in raw_candidates:
                t = cand['metadata']['title']
                if t not in seen_titles:
                    seen_titles.add(t)
                    unique_candidates.append(cand)
            
            debug_print(f"Found {len(unique_candidates)} unique raw candidates from ZIM search")
            
            # Joint 2: Article Scoring
            if self.use_joints and self.scorer_joint and entity_info:
                 debug_print("0c. Scoring candidates with Joint 2...")
                 titles_to_score = [c['metadata']['title'] for c in unique_candidates[:20]]
                 scored_titles_list = self.scorer_joint.score(query, entity_info, titles_to_score)
                 
                 scored_map = {t: s for t, s in scored_titles_list}
                 
                 final_candidates = []
                 for cand in unique_candidates:
                     t = cand['metadata']['title']
                     if t in scored_map:
                         cand['score'] = scored_map[t]
                         if cand['score'] >= 4.0:
                             final_candidates.append(cand)
                             
                 final_candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
                 candidates_to_index = final_candidates[:5]
                 
                 debug_print(f"Joint 2 selected {len(candidates_to_index)} articles for indexing")
                 
                 # Joint 2.5: Coverage Verification
                 if self.coverage_joint and is_comparison:
                      debug_print("0d. Verifying coverage (Joint 2.5)...")
                      coverage = self.coverage_joint.verify_coverage(entity_info, candidates_to_index)
                      
                      if not coverage['complete']:
                           debug_print(f"Coverage Gap Detected! Missing: {coverage['missing']}")
                           debug_print(f"Triggering gap-fill search for: {coverage['suggested_searches']}")
                           for gap_term in coverage['suggested_searches']:
                                gap_results = self.search_by_title(gap_term, full_text=True)
                                for res in gap_results:
                                     if res['metadata']['title'] not in seen_titles:
                                          res['score'] = 10.0
                                          candidates_to_index.append(res)
                                          seen_titles.add(res['metadata']['title'])
                                          break
                           debug_print(f"After gap-fill: {len(candidates_to_index)} candidates")
            else:
                 candidates_to_index = unique_candidates[:5]

            # 0e. Indexing Loop
            newly_indexed = 0
            if self.faiss_index is None:
                 debug_print("Creating new JIT FAISS index...")
                 self.faiss_index = faiss.IndexFlatIP(384)
            
            for cand in candidates_to_index:
                path = cand['metadata']['path']
                title = cand['metadata']['title']
                text = cand['text']
                
                if path in self.indexed_paths:
                    continue
                    
                debug_print(f"JIT Indexing: '{title}' ({len(text)} chars)...")
                raw_chunks = TextProcessor.chunk_text(text, chunk_size=500, overlap=50)
                
                if not raw_chunks:
                    continue
                    
                embeddings = self.encoder.encode(raw_chunks)
                faiss.normalize_L2(embeddings)
                self.faiss_index.add(embeddings)
                
                for i, chunk_text in enumerate(raw_chunks):
                    self.doc_chunks.append(chunk_text)
                    self.documents.append({
                        'title': title,
                        'path': path,
                        'source_zim': cand['metadata'].get('source_zim'),
                        'chunk_id': self._chunk_id
                    })
                    self._chunk_id += 1
                
                self.indexed_paths.add(path)
                newly_indexed += 1
            
            debug_print(f"JIT Indexing Complete. Added {newly_indexed} articles. Total chunks: {len(self.doc_chunks)}")
            
        except Exception as e:
            print(f"JIT Indexing Failed: {e}")
            import traceback
            traceback.print_exc()

        # 1. Dense Retrieval
        debug_print("-" * 70)
        debug_print("PHASE 1: DENSE RETRIEVAL")
        
        if not self.faiss_index or self.faiss_index.ntotal == 0:
            print("No index available for retrieval.")
            return []

        q_emb = self.encoder.encode([query])
        faiss.normalize_L2(q_emb)
        
        search_k = top_k * 3 
        D, I = self.faiss_index.search(q_emb, search_k)
        
        dense_results = []
        for i, idx in enumerate(I[0]):
            if idx == -1 or idx >= len(self.doc_chunks):
                continue
            dense_results.append((idx, float(D[0][i])))
            
        debug_print(f"Dense search found {len(dense_results)} candidates")

        top_candidates = dense_results 
        
        # 4. Neural Reranking / LLM Filtering (Joint 3)
        debug_print("-" * 70)
        debug_print("PHASE 3: LLM FILTERING (JOINT 3)")
        
        reranked_results = []
        
        if self.use_joints and self.filter_joint:
            candidate_chunks = []
            for idx, score in top_candidates:
                candidate_chunks.append({
                    'text': self.doc_chunks[idx],
                    'metadata': self.documents[idx],
                    'rrf_score': score
                })
            
            debug_print("Filtering chunks with Joint 3...")
            answer_type = entity_info.get('answer_type') if entity_info else None
            
            try:
                filtered_chunks = self.filter_joint.filter(
                    query, 
                    candidate_chunks, 
                    top_k=top_k, 
                    entity_info=entity_info, 
                    mode=mode,
                    answer_type=answer_type
                )
                
                for f_chunk in filtered_chunks:
                     for idx, _ in top_candidates:
                         if self.doc_chunks[idx] == f_chunk['text']:
                             new_score = f_chunk.get('relevance_score', 0)
                             reranked_results.append((idx, new_score))
                             break
                
                debug_print(f"Joint 3 kept {len(reranked_results)} chunks")
                
            except Exception as e:
                debug_print(f"Joint 3 filtering failed: {type(e).__name__}: {e}")
                debug_print("Falling back to neural reranking")
                self.use_joints = False
        
        if not reranked_results:
            debug_print("Using fallback ranking...")
            reranked_results = top_candidates[:top_k]

        # === ADAPTIVE RAG: QUALITY CHECK & REBOUND ===
        if rebound_depth == 0 and self.use_joints and self.entity_joint:
            top_score = reranked_results[0][1] if reranked_results else 0.0
            if len(reranked_results) == 0 or top_score < config.ADAPTIVE_THRESHOLD:
                debug_print(f"[ADAPTIVE] Quality Check Failed: top_score={top_score}")
                pass 

        # Joint 3.5: Comparison Synthesis
        comparison_data = None
        if self.use_joints and self.comparison_joint and reranked_results:
             is_comparison = entity_info.get('is_comparison', False) if entity_info else False
             if is_comparison:
                  debug_print("-" * 70)
                  debug_print("PHASE 3.5: COMPARISON SYNTHESIS")
                  joint_chunks = []
                  for idx, score in reranked_results:
                       joint_chunks.append({
                           'text': self.doc_chunks[idx],
                           'relevance_score': score
                       })
                  
                  entities = entity_info.get('entities', []) if entity_info else []
                  entity_names = [e.get('name') for e in entities if e.get('name')]
                  dimension = entity_info.get('comparison_dimension') if entity_info else None
                  
                  try:
                      comparison_data = self.comparison_joint.synthesize_comparison(query, entity_names, dimension, joint_chunks)
                  except Exception as e:
                      debug_print(f"Comparison synthesis failed: {e}")

        # 5. Fact Refinement (Joint 4)
        debug_print("-" * 70)
        debug_print("PHASE 5: FACT REFINEMENT (JOINT 4)")
        extracted_facts = []

        results = []
        debug_print("-" * 70)
        debug_print("PHASE 6: RESULTS ASSEMBLY")
        for i, (idx, score) in enumerate(reranked_results):
            if idx < len(self.doc_chunks):
                doc = self.documents[idx]
                results.append({
                    'text': self.doc_chunks[idx],
                    'metadata': doc,
                    'score': score,
                    'search_context': {
                        'entities': search_terms, 
                        'facts': extracted_facts,
                        'comparison_data': comparison_data,
                        'answer_type': answer_type if 'answer_type' in locals() else None 
                    }
                })
        
        debug_print(f"RETRIEVE COMPLETE. Returning {len(results)} results")
        debug_print("=" * 70)
        return results

    def search_by_title(self, query: str, zim_path: str = None, full_text: bool = False) -> List[Dict]:
        """Search for articles by title using Semantic Title Index or ZIM path fallback."""
        import glob
        
        if not zim_path:
            zims = glob.glob("*.zim")
            if not zims:
                print("No ZIM file found.")
                return []
            zim_path = zims[0]
            
        results = []
        try:
            zim = libzim.Archive(zim_path)
            
            # 1. Semantic Title Search (Preferred)
            if self.title_faiss_index and self.title_metadata and self.encoder:
                 q_emb = self.encoder.encode([query])
                 faiss.normalize_L2(q_emb)
                 D, I = self.title_faiss_index.search(q_emb, 20)
                 
                 for i, idx in enumerate(I[0]):
                     if idx != -1 and idx < len(self.title_metadata):
                         meta = self.title_metadata[int(idx)]
                         try:
                             entry = zim.get_entry_by_path(meta['path'])
                             item = entry.get_item()
                             content = item.content.tobytes().decode('utf-8', errors='ignore')
                             results.append({
                                 'text': content,
                                 'metadata': {
                                     'title': meta['title'],
                                     'path': meta['path'],
                                     'source_zim': zim_path
                                 },
                                 'score': float(D[0][i])
                             })
                         except Exception:
                             continue
                 return results
            
            # 2. Heuristic Path Fallback
            guess_title = query.replace(" ", "_")
            paths_to_try = [
                f"A/{guess_title}", 
                f"A/{guess_title.title()}",
                f"A/{query}"
            ]
            
            for p in paths_to_try:
                try:
                    entry = zim.get_entry_by_path(p)
                    if entry:
                        item = entry.get_item()
                        content = item.content.tobytes().decode('utf-8', errors='ignore')
                        results.append({
                             'text': content,
                             'metadata': {
                                 'title': entry.title,
                                 'path': p,
                                 'source_zim': zim_path
                             },
                             'score': 100.0
                        })
                except Exception:
                    pass
            
            return results

        except Exception as e:
            print(f"Search failed: {e}")
            return []

if __name__ == "__main__":
    pass