#!/usr/bin/env bash
set -euo pipefail

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Error: Don't run this script as root. It will use sudo when needed."
    exit 1
fi

# Check sudo access and cache credentials if needed
echo "Checking sudo access..."
if ! sudo -n true 2>/dev/null; then
    echo "This script needs sudo access to remove system components."
    echo "You will be prompted for your password once."
    echo ""
    # Cache sudo credentials
    sudo -v
    # Keep sudo credentials fresh (refresh every 5 minutes)
    (while true; do sleep 300; sudo -v; done) &
    SUDO_PID=$!
    # Kill the background process when script exits
    trap "kill $SUDO_PID 2>/dev/null || true" EXIT
else
    echo "✓ Sudo access confirmed"
fi

# Create Python GUI script for uninstaller
python3 << 'PYTHON_EOF'
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import subprocess
import sys
import os

class UninstallerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kiwix RAG Uninstaller")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # Variables
        self.remove_krag = tk.BooleanVar(value=True)
        self.remove_python_packages = tk.BooleanVar(value=True)
        self.remove_ollama = tk.BooleanVar(value=True)
        self.remove_kiwix = tk.BooleanVar(value=True)
        self.remove_model = tk.BooleanVar(value=False)  # Default: keep model
        
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title_label = tk.Label(
            self.root,
            text="Kiwix RAG Uninstaller",
            font=("Arial", 16, "bold"),
            pady=10
        )
        title_label.pack()
        
        # Description
        desc_label = tk.Label(
            self.root,
            text="Select components to remove (installed by setup.sh):",
            font=("Arial", 10),
            pady=5
        )
        desc_label.pack()
        
        # Checkboxes frame
        checkboxes_frame = ttk.Frame(self.root, padding=20)
        checkboxes_frame.pack(fill=tk.BOTH, expand=True)
        
        # Checkboxes
        ttk.Checkbutton(
            checkboxes_frame,
            text="Remove 'krag' command (/usr/local/bin/krag)",
            variable=self.remove_krag
        ).pack(anchor=tk.W, pady=5)
        
        ttk.Checkbutton(
            checkboxes_frame,
            text="Remove Python packages (requests, sentence-transformers, chromadb, tiktoken)",
            variable=self.remove_python_packages
        ).pack(anchor=tk.W, pady=5)
        
        ttk.Checkbutton(
            checkboxes_frame,
            text="Remove Ollama (system installation)",
            variable=self.remove_ollama
        ).pack(anchor=tk.W, pady=5)
        
        ttk.Checkbutton(
            checkboxes_frame,
            text="Remove Kiwix tools (kiwix-tools package)",
            variable=self.remove_kiwix
        ).pack(anchor=tk.W, pady=5)
        
        ttk.Checkbutton(
            checkboxes_frame,
            text="Remove downloaded model (llama3.2:1b) - ~1.3GB",
            variable=self.remove_model
        ).pack(anchor=tk.W, pady=5)
        
        # Note
        note_label = tk.Label(
            checkboxes_frame,
            text="Note: Project directory, ZIM files, and RAG indexes will NOT be removed.",
            font=("Arial", 9),
            fg="gray",
            wraplength=550
        )
        note_label.pack(pady=10)
        
        # Buttons frame
        buttons_frame = ttk.Frame(self.root, padding=10)
        buttons_frame.pack(fill=tk.X)
        
        ttk.Button(
            buttons_frame,
            text="Uninstall Selected",
            command=self.start_uninstall
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame,
            text="Cancel",
            command=self.root.quit
        ).pack(side=tk.RIGHT, padx=5)
        
        # Status text area
        self.status_text = scrolledtext.ScrolledText(
            self.root,
            height=8,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def log(self, message):
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
        self.root.update()
        
    def run_command(self, cmd, description, needs_sudo=False):
        """Run a command and log the result."""
        try:
            self.log(f"  {description}...")
            if needs_sudo:
                # Refresh sudo credentials if needed (non-interactive check)
                refresh_result = subprocess.run(
                    ["sudo", "-v"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if refresh_result.returncode != 0:
                    self.log(f"  ⚠ Sudo credentials expired. Please run the script again.")
                    return False
                
                # Use -n flag to prevent password prompt (credentials should be cached)
                result = subprocess.run(
                    ["sudo", "-n"] + cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30
                )
            else:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30
                )
            self.log(f"  ✓ {description} completed")
            return True
        except subprocess.TimeoutExpired:
            self.log(f"  ✗ {description} timed out")
            return False
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() or e.stdout.strip() or "Unknown error"
            if "password" in error_msg.lower() or "sudo" in error_msg.lower():
                self.log(f"  ✗ {description} failed: Sudo password required. Please run the script again.")
            else:
                self.log(f"  ✗ {description} failed: {error_msg}")
            return False
        except Exception as e:
            self.log(f"  ✗ {description} error: {str(e)}")
            return False
    
    def detect_pip_method(self):
        """Detect if packages were installed with --break-system-packages or --user."""
        # Try to import and check location
        try:
            import chromadb
            import sentence_transformers
            
            # Check if in user site-packages
            import site
            user_site = site.getusersitepackages()
            chromadb_path = chromadb.__file__
            
            if user_site and chromadb_path.startswith(user_site):
                return "--user"
            else:
                return "--break-system-packages"
        except:
            # Default to trying both methods
            return "auto"
    
    def start_uninstall(self):
        """Start the uninstallation process."""
        # Check if anything is selected
        if not any([
            self.remove_krag.get(),
            self.remove_python_packages.get(),
            self.remove_ollama.get(),
            self.remove_kiwix.get(),
            self.remove_model.get()
        ]):
            messagebox.showwarning("Nothing Selected", "Please select at least one component to remove.")
            return
        
        # Confirm model removal separately
        if self.remove_model.get():
            response = messagebox.askyesno(
                "Confirm Model Removal",
                "Are you sure you want to remove the llama3.2:1b model?\n\n"
                "This will delete ~1.3GB of data from ~/.ollama/models/\n"
                "You will need to download it again if you reinstall.\n\n"
                "Remove the model?",
                icon=messagebox.WARNING
            )
            if not response:
                self.remove_model.set(False)
        
        # Final confirmation
        components = []
        if self.remove_krag.get():
            components.append("krag command")
        if self.remove_python_packages.get():
            components.append("Python packages")
        if self.remove_ollama.get():
            components.append("Ollama")
        if self.remove_kiwix.get():
            components.append("Kiwix tools")
        if self.remove_model.get():
            components.append("llama3.2:1b model")
        
        response = messagebox.askyesno(
            "Confirm Uninstallation",
            f"Are you sure you want to remove:\n\n" + "\n".join(f"  • {c}" for c in components) + "\n\n"
            "This action cannot be undone.",
            icon=messagebox.WARNING
        )
        
        if not response:
            return
        
        # Disable buttons during uninstall
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Button):
                        child.config(state=tk.DISABLED)
        
        self.log("=== Starting Uninstallation ===")
        self.log("")
        
        success_count = 0
        total_count = 0
        
        # Remove krag command
        if self.remove_krag.get():
            total_count += 1
            if os.path.exists("/usr/local/bin/krag"):
                if self.run_command(
                    ["rm", "-f", "/usr/local/bin/krag"],
                    "Removing krag command",
                    needs_sudo=True
                ):
                    success_count += 1
            else:
                self.log("  krag command not found (may already be removed)")
                success_count += 1
        
        # Remove Python packages
        if self.remove_python_packages.get():
            total_count += 1
            packages = ["requests", "sentence-transformers", "chromadb", "tiktoken"]
            
            # pip uninstall automatically detects where packages were installed
            # No need for --user or --break-system-packages flags (those are install-only)
            cmd = ["python3", "-m", "pip", "uninstall", "-y"] + packages
            
            if self.run_command(cmd, "Removing Python packages", needs_sudo=False):
                success_count += 1
        
        # Remove Ollama
        if self.remove_ollama.get():
            total_count += 1
            # Check if Ollama was installed via install script (usually in /usr/local/bin)
            if os.path.exists("/usr/local/bin/ollama"):
                # Ollama install script typically installs to /usr/local/bin
                if self.run_command(
                    ["rm", "-f", "/usr/local/bin/ollama"],
                    "Removing Ollama binary",
                    needs_sudo=True
                ):
                    success_count += 1
            else:
                self.log("  Ollama binary not found in /usr/local/bin")
                self.log("  Note: If Ollama was installed via package manager, remove it manually")
                success_count += 1
        
        # Remove Kiwix tools
        if self.remove_kiwix.get():
            total_count += 1
            if self.run_command(
                ["apt", "remove", "-y", "kiwix-tools"],
                "Removing Kiwix tools package",
                needs_sudo=True
            ):
                success_count += 1
        
        # Remove model
        if self.remove_model.get():
            total_count += 1
            model_path = os.path.expanduser("~/.ollama/models")
            
            # Find ollama command
            ollama_cmd = None
            if subprocess.run(["which", "ollama"], capture_output=True).returncode == 0:
                ollama_cmd = "ollama"
            elif os.path.exists("/usr/local/bin/ollama"):
                ollama_cmd = "/usr/local/bin/ollama"
            elif os.path.exists("/usr/bin/ollama"):
                ollama_cmd = "/usr/bin/ollama"
            
            if ollama_cmd and os.path.exists(model_path):
                # Remove the specific model
                if self.run_command(
                    [ollama_cmd, "rm", "llama3.2:1b"],
                    "Removing llama3.2:1b model",
                    needs_sudo=False
                ):
                    success_count += 1
            elif not ollama_cmd:
                self.log("  Ollama command not found (may already be removed)")
                # Try to remove model directory directly
                if os.path.exists(model_path):
                    self.log("  Attempting to remove model directory directly...")
                    import shutil
                    try:
                        shutil.rmtree(model_path)
                        self.log("  ✓ Model directory removed")
                        success_count += 1
                    except Exception as e:
                        self.log(f"  ✗ Failed to remove model directory: {e}")
                else:
                    self.log("  Model directory not found (may already be removed)")
                    success_count += 1
            else:
                self.log("  Model directory not found (may already be removed)")
                success_count += 1
        
        # Summary
        self.log("")
        self.log("=== Uninstallation Complete ===")
        if success_count == total_count:
            self.log(f"✓ Successfully removed {success_count}/{total_count} components")
            messagebox.showinfo("Uninstallation Complete", f"Successfully removed {success_count} component(s).")
        else:
            self.log(f"⚠ Partially completed: {success_count}/{total_count} components removed")
            messagebox.showwarning(
                "Uninstallation Complete",
                f"Uninstallation completed with some issues.\n\n"
                f"Removed: {success_count}/{total_count} components\n\n"
                f"Check the log above for details."
            )
        
        # Re-enable cancel button
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Button) and child.cget("text") == "Cancel":
                        child.config(state=tk.NORMAL, text="Close", command=self.root.quit)

if __name__ == "__main__":
    root = tk.Tk()
    app = UninstallerGUI(root)
    root.mainloop()
PYTHON_EOF

