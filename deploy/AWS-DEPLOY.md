# AWS Deployment Guide (Free Tier / Student Account)

Hands-on walkthrough to deploy FinSight-RAG on AWS using your student account.

## 1. Set Up AWS Account

1. Sign in to https://aws.amazon.com/student (or main console with your student-tier account)
2. Choose region closest to you (e.g., `us-east-1`)
3. Make sure IAM is set up: top-right → click your name → "My Security Credentials" → enable MFA (free)

## 2. Create IAM Role for EC2 (so it can read Secrets Manager + write CloudWatch)

1. Console → **IAM** → **Roles** → **Create role**
2. Trusted entity: **AWS service** → **EC2**
3. Attach policies:
   - `AmazonS3FullAccess` (or restrict to your bucket later)
   - `AmazonRDSFullAccess` (or `AmazonRDSReadOnly` for reads)
   - `SecretsManagerReadWrite`
   - `CloudWatchAgentServerPolicy`
4. Name: `finsight-ec2-role`
5. Save role ARN (looks like `arn:aws:iam::123456789012:role/finsight-ec2-role`)

## 3. Create RDS PostgreSQL (free tier)

1. Console → **RDS** → **Create database**
2. Engine: **PostgreSQL** (latest 15+)
3. Template: **Free tier**
4. DB instance identifier: `finsight-db`
5. Master username: `finsight_admin`
6. Master password: click **Auto generate** → **copy and save** somewhere safe
7. DB instance class: db.t3.micro (free tier)
8. Storage: 20 GiB gp2 (free tier)
9. **Public access: No** (security)
10. VPC: default (or create a new VPC)
11. Initial database name: `finsight`
12. **Create database** — wait 5–10 min for "Available"

**Get the endpoint:** RDS → `finsight-db` → **Connectivity & security** → copy **Endpoint** (e.g., `finsight-db.xxxxx.us-east-1.rds.amazonaws.com`)

## 4. Create Secrets in Secrets Manager

1. Console → **Secrets Manager** → **Store a new secret**
2. **Other type of secret** (plaintext)
3. Paste JSON:

```json
{
  "JWT_SECRET": "paste-64-char-random-string-here-use-openssl-rand-hex-32",
  "GEMINI_API_KEY": "paste-your-real-gemini-key",
  "DB_PASSWORD": "paste-the-rds-master-password-from-step-3"
}
```

4. Name: `finsight/prod`
5. **Store** → copy the **Secret ARN**

Generate JWT_SECRET locally:
```bash
openssl rand -hex 32
```

## 5. Create S3 Bucket (for backups)

1. Console → **S3** → **Create bucket**
2. Name: `finsight-backups-yourname` (must be globally unique)
3. Region: same as EC2
4. **Block Public Access: ON** (default)
5. Create

## 6. Launch EC2 Instance (free tier)

1. Console → **EC2** → **Launch instance**
2. Name: `finsight-server`
3. AMI: **Amazon Linux 2023** (free tier eligible)
4. Instance type: **t2.micro** (free tier) or t3.micro
5. Key pair: create new `finsight-key`, **download .pem file** — keep it safe
6. Network settings:
   - **Auto-assign public IP: Enable**
   - Security group: create new
   - Add rule: **HTTP (80)**, source 0.0.0.0/0
   - Add rule: **HTTPS (443)**, source 0.0.0.0/0
   - Add rule: **SSH (22)**, source **My IP** only
7. **Advanced details**:
   - IAM instance profile: `finsight-ec2-role` (from step 2)
   - User data: paste from `deploy/ec2_user_data.sh` in this repo
8. **Launch instance** — wait 2-3 min for "Running" state

## 7. Connect to EC2

On Windows (PowerShell):
```powershell
ssh -i C:\path\to\finsight-key.pem ec2-user@<EC2-PUBLIC-IP>
```

On Mac/Linux:
```bash
chmod 400 finsight-key.pem
ssh -i ./finsight-key.pem ec2-user@<EC2-PUBLIC-IP>
```

## 8. Set Up App on EC2

The user-data script installs Docker, but you still need to:

```bash
# Clone your repo
cd ~
git clone https://github.com/ThrineshAV/financial-rag.git
cd financial-rag

# Pull secrets from Secrets Manager and save to .env
aws secretsmanager get-secret-value --secret-id finsight/prod --region us-east-1 \
  | python -c "import sys, json; d=json.load(sys.stdin); print(json.dumps(json.loads(d['SecretString']), indent=2))" \
  > .env

# Append database URL
echo "DATABASE_URL=postgresql://finsight_admin:$(aws secretsmanager get-secret-value --secret-id finsight/prod --region us-east-1 --query SecretString --output text | python -c 'import sys,json; print(json.load(sys.stdin)[\"DB_PASSWORD\"])')@<RDS-ENDPOINT>:5432/finsight" >> .env
echo "USE_AWS_SECRETS=true" >> .env

# Build and run
docker build -t finsight-rag .
docker run -d --name finsight -p 80:8000 --env-file .env finsight-rag
```

## 9. Verify It's Running

From anywhere:
```bash
curl http://<EC2-PUBLIC-IP>/health
# Should return: {"status":"ok"}
```

## 10. Set Up CloudWatch (Monitoring)

1. Console → **CloudWatch** → **Logs** → **Log groups** → **Create log group**: `/finsight/app`
2. EC2 → select instance → **Actions** → **Monitor and troubleshoot** → **Manage CloudWatch alarms**
3. Create alarm: CPU > 80% for 5 min → send email (set up SNS topic)

## 11. Backup S3 (manual or scheduled)

Upload vector_db and auth DB backup to S3:

```bash
# From EC2
aws s3 cp /home/ec2-user/financial-rag/vector_db s3://finsight-backups-yourname/vector_db/ --recursive
```

Schedule via cron (on EC2):
```bash
crontab -e
# Add: 0 2 * * * aws s3 cp /home/ec2-user/financial-rag/vector_db s3://finsight-backups-yourname/vector_db/$(date +\%F)/ --recursive
```

## 12. Resume Lines You Can Claim

- "Deployed financial RAG system on AWS free tier (EC2, RDS, S3, Secrets Manager, CloudWatch)"
- "Migrated SQLite auth to RDS PostgreSQL with Secrets Manager integration"
- "Containerized FastAPI with Docker; production WSGI behind nginx (optional)"

## What's Free (Student Tier)

- EC2 t2.micro: 750 hrs/month (12 months)
- RDS db.t3.micro: 750 hrs/month (12 months)
- S3: 5 GB storage
- Secrets Manager: 1 secret (free), then $0.40/secret/month
- CloudWatch: 10 custom metrics, basic logs

## Stop Everything (Avoid Charges)

1. EC2 → **Stop instance** (still free if within free-tier hours)
2. RDS → leave running (free within 12 months)
3. S3 → leave bucket (5GB free)

To terminate permanently:
- EC2 → **Terminate instance**
- RDS → **Delete** (takes a final snapshot optionally)
- S3 → **Delete bucket**

## Common Issues

**"RDS connection refused"**: check Security Group — RDS needs an inbound rule allowing port 5432 from EC2's security group, not 0.0.0.0/0

**"Secrets Manager access denied"**: EC2 IAM role missing `SecretsManagerReadWrite`

**"EC2 not accessible"**: check Security Group inbound rules (port 80, 22 from your IP)

**"Gemini 401"**: invalid `GEMINI_API_KEY` in Secrets Manager — update it via console

**"Health check fails on port 80"**: app runs on 8000 inside container; need `-p 80:8000` in docker run

## Optional: Add HTTPS (Free via Let's Encrypt)

```bash
# On EC2
sudo yum install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```
