#!/bin/bash

# FinSight-RAG EC2 User Data Script
# Runs on first EC2 boot to install dependencies, clone repo, configure secrets, and start Docker

set -euo pipefail

# Update system
apt-get update -y
apt-get install -y curl git python3 python3-pip python3-venv ca-certificates

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
echo "Docker installed"

# Create app directory and clone repository
mkdir -p /app/financial-rag
cd /app/financial-rag
git clone https://github.com/ThrineshAV/financial-rag.git .
git checkout main
git pull origin main

# Install Python dependencies
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Install AWS CLI
pip install awscli

# Copy Secrets Manager setup script
shared_dir="/home/ec2-user/shared"
mkdir -p $shared_dir

# Create .env file from AWS Secrets Manager
aws secretsmanager get-secret-value --secret-id finsight/prod --region us-east-1 \
  --query SecretString --output text | python3 -c "
import sys, json
data = json.load(sys.stdin)
# Filter out DB_PASSWORD (stored separately)
db_password = data.get('DB_PASSWORD', '')
if db_password:
    data['DB_PASSWORD'] = '**REDACTED**'
print(json.dumps(data, indent=2))
" > .env

# Append DATABASE_URL
echo "DATABASE_URL=postgresql://finsight_admin:<REPLACE_WITH_DB_PASSWORD>@<RDS_ENDPOINT>:5432/finsight" >> .env

# Replace placeholders with actual DB password from AWS Secrets Manager
DB_PASSWORD=$(aws secretsmanager get-secret-value --secret-id finsight/prod --region us-east-1 --query SecretString --output text | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('DB_PASSWORD', ''))" | tr -d '\r')
DB_ENDPOINT=$(aws ec2 describe-instances --filters Name=tag:Name,Values=finsight-db --query 'Reservations[0].Instances[0].PublicIpAddress' --output text --region us-east-1)

# Update .env with real values
sed -i "s|<REPLACE_WITH_DB_PASSWORD>|$DB_PASSWORD|g" .env
sed -i "s|<RDS_ENDPOINT>|$DB_ENDPOINT|g" .env

# Build Docker image
docker build -t finsight-rag .

# Create systemd service for Docker container
sudo bash -c 'cat > /etc/systemd/system/finsight-rag.service <<EOF
[Unit]
Description=FinSight-RAG API Service
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/app/financial-rag
Environment="PYTHONPATH=/app/financial-rag"
EnvironmentFile=/app/financial-rag/.env
ExecStart=/usr/bin/docker run --name finsight-rag --restart unless-stopped -p 8000:8000 -v /app/financial-rag/vector_db:/app/vector_db -v /app/financial-rag/data:/app/data -d finsight-rag
ExecStop=/usr/bin/docker stop finsight-rag
ExecStartPost=/usr/bin/docker cp finsight-rag:/app/financial-rag/vector_db /app/financial-rag/vector_db 2>/dev/null || true

[Install]
WantedBy=multi-user.target
EOF'

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable finsight-rag
sudo systemctl restart finsight-rag

# Install CloudWatch Agent for production logging
curl -sL https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb -o /tmp/amazon-cloudwatch-agent.deb
sudo dpkg -i /tmp/amazon-cloudwatch-agent.deb || sudo apt-get install -f -y

# Configure CloudWatch Agent
sudo mkdir -p /opt/aws/amazon-cloudwatch-agent/etc
sudo bash -c 'cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json <<EOF
{
  "agent": {
    "metrics_collection_interval": 60,
    "run_as_user": "root"
  },
  "logs": {
    "metrics_collected": {
      "mem": {
        "measurement": ["mem_used_percent"]
      }
    },
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/finsight-rag/*.log",
            "log_group_name": "/finsight/app",
            "log_stream_name": "{instance_id}"
          }
        ]
      }
    }
  }
}
EOF'

# Start CloudWatch Agent
sudo systemctl start amazon-cloudwatch-agent || sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s

# Install curl for health checks
cat > /usr/local/bin/healthcheck.sh <<'EOS'
#!/bin/bash
if curl -f http://localhost:8000/health > /dev/null; then
    echo "OK"
    exit 0
else
    echo "UNHEALTHY"
    exit 1
fi
EOS
chmod +x /usr/local/bin/healthcheck.sh

# Log completion
echo "FinSight-RAG setup completed successfully"
echo "You can check the service status with: systemctl status finsight-rag"
echo "Health check: /usr/local/bin/healthcheck.sh"
echo "API is available at http://<EC2_PUBLIC_IP>:8000"