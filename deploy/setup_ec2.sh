#!/bin/bash
# MLCopilot AI — EC2 Setup Script
# ================================
# Run this ONCE on a fresh Amazon Linux 2023 / Ubuntu EC2 instance.
# Installs Docker, pulls the app, and starts everything.
#
# Usage:
#   chmod +x deploy/setup_ec2.sh
#   ./deploy/setup_ec2.sh
#
# Prerequisites:
#   - EC2 instance (t3.micro or t3.small, Amazon Linux 2023 or Ubuntu 22.04)
#   - Security group with ports 22, 80, 443, 8000 open
#   - Your .env file copied to the instance

set -e  # exit on any error

echo "=============================================="
echo "  MLCopilot AI — EC2 Setup"
echo "=============================================="

# ── 1. Detect OS and install Docker ───────────────────────────────────────────
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
fi

if [ "$OS" = "amzn" ]; then
    echo "[1/6] Installing Docker (Amazon Linux)..."
    sudo yum update -y
    sudo yum install -y docker git
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo usermod -aG docker ec2-user
elif [ "$OS" = "ubuntu" ]; then
    echo "[1/6] Installing Docker (Ubuntu)..."
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg git
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
         https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
         | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    sudo usermod -aG docker ubuntu
fi

# ── 2. Install Docker Compose ─────────────────────────────────────────────────
echo "[2/6] Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
    -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# ── 3. Install Nginx ──────────────────────────────────────────────────────────
echo "[3/6] Installing Nginx..."
if [ "$OS" = "amzn" ]; then
    sudo yum install -y nginx
else
    sudo apt-get install -y nginx
fi
sudo systemctl enable nginx

# ── 4. Clone / update repo ────────────────────────────────────────────────────
echo "[4/6] Cloning repository..."
REPO_DIR="/home/ec2-user/mlcopilot-ai"
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR" && git pull
else
    git clone https://github.com/YOUR_USERNAME/MLCopilot-AI.git "$REPO_DIR"
    cd "$REPO_DIR"
fi

# Copy .env if it doesn't exist yet
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "⚠️  Edit /home/ec2-user/mlcopilot-ai/.env and fill in your AWS / LLM credentials."
    echo "   Then run: docker-compose up -d"
    echo ""
fi

# ── 5. Copy Nginx config ──────────────────────────────────────────────────────
echo "[5/6] Configuring Nginx..."
sudo cp deploy/nginx.conf /etc/nginx/conf.d/mlcopilot.conf
sudo nginx -t && sudo systemctl restart nginx

# ── 6. Build and start containers ────────────────────────────────────────────
echo "[6/6] Building and starting containers..."
docker-compose up --build -d

echo ""
echo "=============================================="
echo "  ✅  MLCopilot AI is running!"
echo "  API  → http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000"
echo "  Docs → http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000/docs"
echo "=============================================="
