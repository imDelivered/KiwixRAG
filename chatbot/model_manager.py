"""
Model Manager for Local Inference.
Handles downloading GGUF models from Hugging Face and loading them via llama-cpp-python.
"""

import os
import sys
import glob
from typing import Optional, Dict, List
from huggingface_hub import hf_hub_download, list_repo_files
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None
    print("WARNING: llama-cpp-python not installed. Local inference will fail.")

from chatbot import config

class ModelManager:
    """Singleton manager for local LLM models."""
    
    _instances: Dict[str, 'Llama'] = {}
    
    @staticmethod
    def ensure_model_path(repo_id: str) -> str:
        """
        Ensure the model exists locally. varying quantization support.
        Downloads the best available GGUF if not found.
        """
        # Determine path relative to this file (chatbot/model_manager.py -> project_root/shared_models)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_dir = os.path.join(project_root, "shared_models")
        os.makedirs(model_dir, exist_ok=True)
        
        # 1. Check if we already have a suitable file for this repo
        # We store them as "RepoName-Quant.gguf" or just rely on huggingface cache
        # Ideally, we copy/symlink to data/models for clarity, or just use the cache path.
        # Using cache path is safer for updates.
        
        print(f"Checking model availability for: {repo_id}")
        
        # Search strategy: Q5_K_M > Q4_K_M > Q8_0 > Q4_0
        preferences = ["Q5_K_M", "Q4_K_M", "Q8_0", "Q4_0"]
        
        # 0. Fast Path: Check if we have a matching GGUF in the local dir
        # We search for files containing the repo name (or part of it) and the quant
        existing_files = glob.glob(os.path.join(model_dir, "*.gguf"))
        
        if existing_files:
            # Try to match based on preferences
            for quant in preferences:
                # Find files that look like they belong to this model (heuristic: usually filename has quant)
                # Matches if quant is in filename
                matches = [f for f in existing_files if quant.lower() in f.lower()]
                if matches:
                    # Determine if it matches the repo roughly?
                    # Since we centralized, we might have multiple models.
                    # Simple heuristic: Just pick the first match if we assume we only keep what we want?
                    # Better: Check if the filename roughly matches the repo name's last part
                    repo_name_part = repo_id.split('/')[-1]
                    
                    # Heuristic: check if significant part of repo name is in filename
                    # For DarkIdol: look for "DarkIdol"
                    # For Aletheia: look for "Aletheia" or "Llama-3.2-3B" vs "Llama-3.1-8B"
                    
                    match_found = False
                    candidate_file = None

                    for candidate in matches:
                        if "DarkIdol" in repo_name_part and "DarkIdol" in candidate:
                            match_found = True
                            candidate_file = candidate
                            break
                        elif "Aletheia" in repo_name_part and ("Aletheia" in candidate or "Llama-3.2" in candidate):
                            match_found = True
                            candidate_file = candidate
                            break
                        elif "Llama-3.1" in repo_name_part and "Llama-3.1" in candidate:
                             match_found = True
                             candidate_file = candidate
                             break
                             
                    if match_found and candidate_file:
                        print(f"Found local cached model: {candidate_file}")
                        return candidate_file
                    else:
                        print(f"Skipping ambiguous local file(s) for {repo_id}")
                    
                    if matches and len(existing_files) < 10: # Fallback if few models
                         pass # Could fall through to download logic
                    
                    # If we are here, we didn't return a match. Continue to next quant preference or download.

        # List files in repo
        try:
            files = list_repo_files(repo_id)
            gguf_files = [f for f in files if f.endswith('.gguf')]
            
            if not gguf_files:
                raise ValueError(f"No GGUF files found in {repo_id}")
            
            selected_file = None
            for quant in preferences:
                matches = [f for f in gguf_files if quant.lower() in f.lower()]
                if matches:
                    selected_file = matches[0]
                    # Prefer "uncensored" in name if duplicates exist? 
                    # Usually repo is specific enough.
                    break
            
            if not selected_file:
                # Fallback to the smallest/first
                selected_file = gguf_files[0]
                
            print(f"Selected model file: {selected_file}")
            
            # Download
            path = hf_hub_download(repo_id=repo_id, filename=selected_file, local_dir=model_dir)
            print(f"Model available at: {path}")
            return path
            
        except Exception as e:
            print(f"Error resolving model {repo_id}: {e}")
            # Final Fallback: Check if ANY file exists in model_dir
            if existing_files:
                 print(f"Network error, falling back to local file: {existing_files[0]}")
                 return existing_files[0]
            raise

    @classmethod
    def get_model(cls, repo_id: str, n_ctx: int = 4096, n_gpu_layers: int = -1) -> 'Llama':
        """
        Get or load a Llama model instance.
        Enforces single-model policy to prevent VRAM OOM.
        """
        if repo_id in cls._instances:
            return cls._instances[repo_id]
            
        # Unload ALL other models to free VRAM before loading new one
        if cls._instances:
            print(f"Unloading {len(cls._instances)} active models to free VRAM for {repo_id}...")
            import gc
            for key in list(cls._instances.keys()):
                print(f"Unloading model: {key}")
                # Explicitly delete the Llama object
                model_instance = cls._instances[key]
                del model_instance
                del cls._instances[key]
            
            # Force garbage collection to ensure VRAM is released
            cls._instances.clear()
            gc.collect()
            
        if Llama is None:
            raise ImportError("llama-cpp-python is missing")
            
        print(f"Loading model: {repo_id}...")
        try:
            model_path = cls.ensure_model_path(repo_id)
            
            # Load with GPU offload
            # n_gpu_layers = -1 means 'all layers' (good for 3060 12GB)
            # n_ctx depends on usage (8192 for darkidol, 2048 for joints)
            
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=False
            )
            
            cls._instances[repo_id] = llm
            print(f"Model {repo_id} loaded successfully.")
            return llm
        except Exception as e:
            print(f"Failed to load model {repo_id}: {e}")
            raise

    @classmethod
    def close_all(cls):
        """Free memory."""
        cls._instances.clear()
        # Python GC should handle the rest if no refs remain
