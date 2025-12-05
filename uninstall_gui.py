
import tkinter as tk
from tkinter import ttk, messagebox
import os
import shutil
import subprocess

class UninstallGUI:
    def __init__(self):
        try:
            self.root = tk.Tk()
        except:
             print("Tkinter not available. Use CLI.")
             return

        self.root.title("KiwixRAG Uninstaller")
        self.root.geometry("400x450")
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Header
        header = ttk.Label(self.root, text="Uninstall Options", font=("Arial", 16, "bold"))
        header.pack(pady=20)
        
        # Checkbox variables
        self.var_venv = tk.BooleanVar(value=True)
        self.var_indices = tk.BooleanVar(value=True)
        self.var_ollama = tk.BooleanVar(value=False)
        self.var_cache = tk.BooleanVar(value=True)
        
        # Options Frame
        opts_frame = ttk.Frame(self.root)
        opts_frame.pack(fill=tk.BOTH, expand=True, padx=40)
        
        ttk.Checkbutton(opts_frame, text="Remove Virtual Env (venv)", variable=self.var_venv).pack(anchor=tk.W, pady=5)
        ttk.Checkbutton(opts_frame, text="Remove Search Indices (data/)", variable=self.var_indices).pack(anchor=tk.W, pady=5)
        ttk.Checkbutton(opts_frame, text="Remove Cache (__pycache__)", variable=self.var_cache).pack(anchor=tk.W, pady=5)
        
        # Hazardous options
        ttk.Separator(opts_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        ttk.Checkbutton(opts_frame, text="Delete AI Model (llama3.2:1b)", variable=self.var_ollama).pack(anchor=tk.W, pady=5)
        
        # Uninstall Button
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=30)
        
        ttk.Button(btn_frame, text="Uninstall Selected", command=self.confirm_and_run).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Cancel", command=self.root.quit).pack(side=tk.LEFT, padx=10)
        
    def confirm_and_run(self):
        to_delete = []
        if self.var_venv.get(): to_delete.append("Virtual Environment")
        if self.var_indices.get(): to_delete.append("Search Indices")
        if self.var_cache.get(): to_delete.append("System Caches")
        if self.var_ollama.get(): to_delete.append("Ollama Model")
        
        if not to_delete:
            messagebox.showinfo("Nothing Selected", "No items selected for removal.")
            return

        msg = "Are you sure you want to delete:\n\n" + "\n".join(f"- {item}" for item in to_delete)
        if messagebox.askyesno("Confirm Uninstall", msg, icon='warning'):
            self.perform_uninstall()
            
    def perform_uninstall(self):
        log = []
        try:
            # 1. Venv
            if self.var_venv.get() and os.path.exists("venv"):
                shutil.rmtree("venv")
                log.append("✅ Removed venv")
                
            # 2. Indices
            if self.var_indices.get() and os.path.exists("data"):
                shutil.rmtree("data")
                log.append("✅ Removed data/")

            # 3. Cache
            if self.var_cache.get():
                for root, dirs, files in os.walk("."):
                    for d in dirs:
                        if d == "__pycache__":
                            shutil.rmtree(os.path.join(root, d))
                log.append("✅ Removed __pycache__")

            # 4. ZIM Files - REMOVED: ZIM files are user data and should never be deleted

            # 5. Ollama Model
            if self.var_ollama.get():
                subprocess.run(["ollama", "rm", "llama3.2:1b"], check=False)
                log.append("✅ Removed Ollama model")

            messagebox.showinfo("Success", "\n".join(log))
            self.root.quit()
            
        except Exception as e:
            messagebox.showerror("Error", f"Uninstall failed: {e}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = UninstallGUI()
    app.run()
