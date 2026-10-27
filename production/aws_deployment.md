# ☁️ AWS Deployment Guide — SummarizeAI

> **Enterprise Cloud Deployment Architectures for Text Summarization System**

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Architecture A: AWS EC2 with Nginx](#architecture-a-aws-ec2-with-nginx)
3. [Architecture B: AWS ECR + ECS Fargate](#architecture-b-aws-ecr--ecs-fargate)
4. [Architecture C: AWS App Runner](#architecture-c-aws-app-runner)
5. [Cost Optimization Matrix](#cost-optimization-matrix)
6. [Teardown Commands](#teardown-commands)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    AWS Deployment Options                     │
├──────────────┬───────────────────┬───────────────────────────┤
│   EC2 + Nginx │  ECR + ECS Fargate │      App Runner          │
│  (Full Control)│ (Serverless Containers)│ (Fully Managed)       │
├──────────────┼───────────────────┼───────────────────────────┤
│ Manual scaling │  Auto-scaling      │  Auto-scaling            │
│ SSL via Certbot│  ALB + ACM SSL     │  Built-in SSL            │
│ Lowest cost    │  Pay-per-task      │  Pay-per-request         │
│ Most control   │  Production-grade  │  Simplest setup          │
└──────────────┴───────────────────┴───────────────────────────┘
```

---

## Architecture A: AWS EC2 with Nginx

### Step 1: Launch EC2 Instance

```bash
# Launch Ubuntu 22.04 LTS instance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxx \
  --subnet-id subnet-xxxxxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=summarize-ai}]' \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]'
```

### Step 2: Security Group Configuration

```bash
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxx \
  --protocol tcp --port 22 --cidr YOUR_IP/32

aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxx \
  --protocol tcp --port 80 --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxx \
  --protocol tcp --port 443 --cidr 0.0.0.0/0
```

### Step 3: Install Dependencies on EC2

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER

# Install Nginx
sudo apt install -y nginx certbot python3-certbot-nginx

# Clone and build
git clone https://github.com/mdzaheerjk/End-to-End-NLP-Text-Summarization-with-Hugging-Face-Transformers.git
cd End-to-End-NLP-Text-Summarization-with-Hugging-Face-Transformers
cp .env.example .env
# Edit .env with production values
docker run -d --name summarize-ai -p 8000:8000 --env-file .env --restart unless-stopped summarize-ai:latest

# Verify health and model initialization
curl -i http://localhost:8000/health
curl -s http://localhost:8000/diagnostics | python -m json.tool
```

### Step 4: Nginx Reverse Proxy Configuration

```nginx
# /etc/nginx/sites-available/summarize-ai
server {
    listen 80;
    server_name your-domain.com;
    client_max_body_size 10M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_connect_timeout 10s;
    }

    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }

    location /diagnostics {
        proxy_pass http://127.0.0.1:8000/diagnostics;
    }
}
```

```bash
# Enable site and SSL
sudo ln -s /etc/nginx/sites-available/summarize-ai /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Let's Encrypt SSL
sudo certbot --nginx -d your-domain.com --non-interactive --agree-tos -m your@email.com
```

### Step 5: Systemd Service Daemon

```ini
# /etc/systemd/system/summarize-ai.service
[Unit]
Description=SummarizeAI Text Summarization Service
After=docker.service
Requires=docker.service

[Service]
Type=simple
Restart=always
RestartSec=10
ExecStartPre=-/usr/bin/docker stop summarize-ai
ExecStartPre=-/usr/bin/docker rm summarize-ai
ExecStart=/usr/bin/docker run --name summarize-ai -p 8000:8000 --env-file /home/ubuntu/app/.env --memory=4g --cpus=2.0 summarize-ai:latest
ExecStop=/usr/bin/docker stop summarize-ai

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable summarize-ai
sudo systemctl start summarize-ai
sudo systemctl status summarize-ai
```

---

## Architecture B: AWS ECR + ECS Fargate

### Step 1: Create ECR Repository

```bash
# Create repository
aws ecr create-repository \
  --repository-name summarize-ai \
  --image-scanning-configuration scanOnPush=true \
  --encryption-configuration encryptionType=AES256

# Authenticate Docker with ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag summarize-ai:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/summarize-ai:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/summarize-ai:latest
```

### Step 2: Create ECS Cluster

```bash
aws ecs create-cluster \
  --cluster-name summarize-ai-cluster \
  --capacity-providers FARGATE FARGATE_SPOT \
  --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1
```

### Step 3: Task Definition

```json
{
  "family": "summarize-ai-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "summarize-ai",
      "image": "<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/summarize-ai:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "APP_ENV", "value": "production"},
        {"name": "LOG_LEVEL", "value": "INFO"}
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 10,
        "retries": 3,
        "startPeriod": 120
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/summarize-ai",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

### Step 4: Application Load Balancer & Target Group

```bash
# Create ALB
aws elbv2 create-load-balancer \
  --name summarize-ai-alb \
  --subnets subnet-xxx subnet-yyy \
  --security-groups sg-xxxxxxxx \
  --scheme internet-facing \
  --type application

# Create target group
aws elbv2 create-target-group \
  --name summarize-ai-tg \
  --protocol HTTP \
  --port 8000 \
  --vpc-id vpc-xxxxxxxx \
  --target-type ip \
  --health-check-path /health \
  --health-check-interval-seconds 30 \
  --healthy-threshold-count 2

# Create listener
aws elbv2 create-listener \
  --load-balancer-arn <ALB_ARN> \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=forward,TargetGroupArn=<TG_ARN>
```

### Step 5: Create Fargate Service with Auto-Scaling

```bash
# Create service
aws ecs create-service \
  --cluster summarize-ai-cluster \
  --service-name summarize-ai-service \
  --task-definition summarize-ai-task \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=<TG_ARN>,containerName=summarize-ai,containerPort=8000" \
  --health-check-grace-period-seconds 120

# Configure auto-scaling
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/summarize-ai-cluster/summarize-ai-service \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 1 \
  --max-capacity 10

# CPU-based scaling policy
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --resource-id service/summarize-ai-cluster/summarize-ai-service \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-name cpu-scaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
    },
    "ScaleOutCooldown": 60,
    "ScaleInCooldown": 300
  }'
```

---

## Architecture C: AWS App Runner

### Step 1: Create App Runner Service

```bash
# From ECR image
aws apprunner create-service \
  --service-name summarize-ai \
  --source-configuration '{
    "AuthenticationConfiguration": {
      "AccessRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/AppRunnerECRAccessRole"
    },
    "ImageRepository": {
      "ImageIdentifier": "<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/summarize-ai:latest",
      "ImageRepositoryType": "ECR",
      "ImageConfiguration": {
        "Port": "8000",
        "RuntimeEnvironmentVariables": {
          "APP_ENV": "production",
          "LOG_LEVEL": "INFO"
        }
      }
    },
    "AutoDeploymentsEnabled": true
  }' \
  --instance-configuration '{
    "Cpu": "2 vCPU",
    "Memory": "4 GB"
  }' \
  --health-check-configuration '{
    "Protocol": "HTTP",
    "Path": "/health",
    "Interval": 10,
    "Timeout": 5,
    "HealthyThreshold": 1,
    "UnhealthyThreshold": 5
  }' \
  --auto-scaling-configuration-arn <AUTO_SCALING_CONFIG_ARN>
```

### Step 2: Configure Auto-Scaling

```bash
aws apprunner create-auto-scaling-configuration \
  --auto-scaling-configuration-name summarize-ai-scaling \
  --max-concurrency 50 \
  --min-size 1 \
  --max-size 5
```

---

## Cost Optimization Matrix

| Architecture | Monthly Cost (Est.) | Best For | Scaling | Setup Complexity |
|---|---|---|---|---|
| **EC2 t3.xlarge** | ~$120/mo | Low traffic, full control | Manual | Medium |
| **EC2 Spot Instance** | ~$40/mo | Dev/staging, cost-sensitive | Manual | Medium |
| **ECS Fargate (2 tasks)** | ~$150/mo | Production, auto-scaling | Automatic | High |
| **ECS Fargate Spot** | ~$50/mo | Fault-tolerant workloads | Automatic | High |
| **App Runner (2 vCPU)** | ~$100/mo | Simplest production deploy | Automatic | Low |

> **Recommendation:** Start with **App Runner** for rapid deployment. Migrate to **ECS Fargate** when you need fine-grained networking, service mesh, or multi-container task definitions.

---

## Teardown Commands

```bash
# === EC2 Teardown ===
aws ec2 terminate-instances --instance-ids i-xxxxxxxx

# === ECS Fargate Teardown ===
aws ecs update-service --cluster summarize-ai-cluster --service summarize-ai-service --desired-count 0
aws ecs delete-service --cluster summarize-ai-cluster --service summarize-ai-service --force
aws ecs delete-cluster --cluster summarize-ai-cluster
aws ecr delete-repository --repository-name summarize-ai --force
aws elbv2 delete-load-balancer --load-balancer-arn <ALB_ARN>
aws elbv2 delete-target-group --target-group-arn <TG_ARN>

# === App Runner Teardown ===
aws apprunner delete-service --service-arn <SERVICE_ARN>

# === CloudWatch Logs Cleanup ===
aws logs delete-log-group --log-group-name /ecs/summarize-ai
```

> ⚠️ **Always verify teardown** with `aws ecs list-services`, `aws ec2 describe-instances`, and check your AWS billing dashboard.
