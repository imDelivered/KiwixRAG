# Deploy Key Setup (Easiest Method)

Deploy keys are SSH keys that only work for one repository. Perfect for this use case!

## Step 1: Generate SSH Key (in your VM)

```bash
ssh-keygen -t ed25519 -C "deploy-key-OWRs" -f ~/.ssh/id_ed25519_OWRs
```

Press Enter twice (no passphrase needed, or set one if you prefer).

## Step 2: Display the Public Key

```bash
cat ~/.ssh/id_ed25519_OWRs.pub
```

**Copy the entire output** - it will look like:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... deploy-key-OWRs
```

## Step 3: Add Deploy Key to GitHub

1. Go to: https://github.com/imDelivered/OWRs/settings/keys
2. Click **"Add deploy key"**
3. **Title:** `VM Deploy Key` (or any name you want)
4. **Key:** Paste the public key you copied
5. Check **"Allow write access"** (so you can push)
6. Click **"Add key"**

## Step 4: Configure Git to Use This Key

```bash
cd "/home/dekko/Desktop/public repo"

# Create SSH config for this repo
mkdir -p ~/.ssh
cat >> ~/.ssh/config << 'EOF'
Host github-OWRs
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_OWRs
    IdentitiesOnly yes
EOF
```

## Step 5: Initialize Git and Push

```bash
cd "/home/dekko/Desktop/public repo"

# Initialize git
git init
git add .
git commit -m "Initial commit: Wiki Chat with RAG"

# Add remote using the SSH config host
git remote add origin git@github-OWRs:imDelivered/OWRs.git

# Push
git branch -M main
git push -u origin main
```

That's it! The deploy key will authenticate automatically.

## Test Connection (Optional)

```bash
ssh -T git@github-OWRs
# Should say: "Hi imDelivered/OWRs! You've successfully authenticated..."
```

