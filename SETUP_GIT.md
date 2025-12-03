# Git Setup Guide

Follow these steps to push your code to GitHub.

## Option 1: HTTPS (Easiest - Recommended)

### Step 1: Initialize Git

```bash
cd "/home/dekko/Desktop/public repo"
git init
git add .
git commit -m "Initial commit: Wiki Chat with RAG"
```

### Step 2: Add GitHub Remote

```bash
git remote add origin https://github.com/imDelivered/OWRs.git
```

### Step 3: Push to GitHub

**If you haven't set up authentication yet:**

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate a new token with `repo` permissions
3. Copy the token

Then push:
```bash
git branch -M main
git push -u origin main
```

When prompted for username: enter your GitHub username
When prompted for password: **paste your personal access token** (not your password)

## Option 2: SSH (If you prefer)

### Step 1: Check if you have an SSH key

```bash
ls -la ~/.ssh/id_*.pub
```

### Step 2: If no key exists, generate one

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# Press Enter to accept default location
# Press Enter twice for no passphrase (or set one)
```

### Step 3: Add SSH key to GitHub

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the output, then:
1. Go to GitHub → Settings → SSH and GPG keys
2. Click "New SSH key"
3. Paste the key and save

### Step 4: Test SSH connection

```bash
ssh -T git@github.com
```

### Step 5: Initialize and push

```bash
cd "/home/dekko/Desktop/public repo"
git init
git add .
git commit -m "Initial commit: Wiki Chat with RAG"
git remote add origin git@github.com:imDelivered/OWRs.git
git branch -M main
git push -u origin main
```

## Quick One-Liner (HTTPS)

If you already have a personal access token:

```bash
cd "/home/dekko/Desktop/public repo" && \
git init && \
git add . && \
git commit -m "Initial commit: Wiki Chat with RAG" && \
git remote add origin https://github.com/imDelivered/OWRs.git && \
git branch -M main && \
git push -u origin main
```

Enter your GitHub username and personal access token when prompted.

