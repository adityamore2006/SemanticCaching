# Deploying the cache

First AWS project notes are inline. The short version: you run four commands, and you never copy an access key onto the server.

## What gets created

| Resource | Why |
|---|---|
| DynamoDB table | Durable store for cached answers and their vectors. On-demand billing, so it costs nothing when idle. |
| IAM role + instance profile | How the server authenticates to AWS. See "Security" below, this is the part worth understanding. |
| Security group | Firewall. SSH locked to your IP, port 8000 open for the API. |
| EC2 instance (`t4g.medium`) | Runs everything: FastAPI, the embedding model, the HNSW graph. |
| EBS volume (20 GB) | The instance's disk. Survives stop/start, holds the graph snapshot. |

## Security: why there are no access keys here

This is the one concept worth getting right on a first AWS project.

You have an access key on your laptop (from `aws configure`). It is a long-lived secret: it works until you delete it, and anyone holding it is you. That is fine on a machine you control, and it is exactly what you must **not** copy onto a server.

Instead, the server gets an **IAM role**. AWS hands the instance short-lived credentials through an internal metadata endpoint, rotates them automatically, and boto3 picks them up with no configuration at all. Nothing is stored on disk, so there is no secret to leak, and if the instance is compromised the attacker gets credentials that expire and that only permit what the role allows.

The role here grants exactly two things:

- `GetItem` / `PutItem` / `Scan` on **the one table this stack creates**, not DynamoDB generally.
- `InvokeModel` on **Anthropic models only**, not `bedrock:*`.

That is least privilege: if the app is compromised, the blast radius is one table and one model family.

**So: never put an access key in the code, in an environment variable on the server, or in a committed file.** If you ever do by accident, delete the key in the IAM console immediately rather than trying to scrub it from git history.

## Deploy

**1. Prerequisites.** An EC2 key pair (a one-time thing, creates a `.pem` file you keep private):

```bash
aws ec2 create-key-pair --key-name semantic-cache \
  --query KeyMaterial --output text > ~/.ssh/semantic-cache.pem
chmod 400 ~/.ssh/semantic-cache.pem
```

**2. Create the stack.** `MyIp` locks SSH to your address:

```bash
aws cloudformation deploy \
  --template-file infra/stack.yaml \
  --stack-name semantic-cache \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
      KeyName=semantic-cache \
      MyIp="$(curl -s https://checkip.amazonaws.com)/32"
```

`CAPABILITY_IAM` is you explicitly acknowledging the stack creates an IAM role. AWS requires it so permissions are never granted silently.

**3. Read the outputs:**

```bash
aws cloudformation describe-stacks --stack-name semantic-cache \
  --query 'Stacks[0].Outputs' --output table
```

Note the `InstanceId`, `PublicIp`, and `CacheTableName`.

**4. Set the app up on the instance.** SSH in (or use `aws ssm start-session --target <InstanceId>`, which needs no SSH key or open port at all, since the role already allows it):

```bash
ssh -i ~/.ssh/semantic-cache.pem ec2-user@<PublicIp>

git clone https://github.com/adityamore2006/SemanticCaching.git
cd SemanticCaching
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Downloads the embedding model once, into the image cache on EBS, so
# later boots do not re-download it.
.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

sudo cp infra/semantic-cache.service /etc/systemd/system/
sudo sed -i "s/REPLACE_ME/<CacheTableName>/" /etc/systemd/system/semantic-cache.service
sudo systemctl daemon-reload
sudo systemctl enable --now semantic-cache
journalctl -u semantic-cache -f
```

**5. Verify** from your laptop:

```bash
curl -s -X POST http://<PublicIp>:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "Can I merge two accounts into one?"}'

curl -s -X POST http://<PublicIp>:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "Is it possible to combine my two separate accounts?"}'

curl -s http://<PublicIp>:8000/stats
```

The first is a miss, the second should hit at about `0.94`, matching the local demo.

## Turning it on and off

This is the whole operational surface, and the reason the bill stays near two dollars:

```bash
aws ec2 stop-instances  --instance-ids <InstanceId>   # done for now
aws ec2 start-instances --instance-ids <InstanceId>   # demo time
```

Stopping runs the systemd shutdown, which writes the graph snapshot, so the next start reloads it instead of rebuilding from DynamoDB.

**The public IP changes on every start** unless you attach an Elastic IP (about $3.65/month). Re-read it with:

```bash
aws ec2 describe-instances --instance-ids <InstanceId> \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

## Costs

| | Demo use (~20 hrs/mo) | Left on 24/7 |
|---|---|---|
| EC2 `t4g.medium` | $0.67 | $24.53 |
| EBS 20 GB gp3 | $1.60 | $1.60 |
| DynamoDB on-demand | ~$0.00 | ~$0.00 |
| **Total** | **≈ $2.30/mo** | **≈ $26.10/mo** |

Set a billing alarm before you walk away from this. It is the habit that prevents surprise bills:

```bash
aws budgets create-budget --account-id "$(aws sts get-caller-identity --query Account --output text)" \
  --budget '{"BudgetName":"monthly","BudgetLimit":{"Amount":"10","Unit":"USD"},"TimeUnit":"MONTHLY","BudgetType":"COST"}'
```

## Tearing it all down

```bash
aws cloudformation delete-stack --stack-name semantic-cache
```

Deletes the instance, volume, role, security group, and table. Nothing is left billing. That single-command teardown is the main practical argument for defining this as a stack instead of clicking it together in the console.

## Bedrock

The miss path needs a Claude model, and on a new account the quota starts at zero. Request an increase for **Claude Haiku 4.5**:

- `L-CCA5DF70` (cross-region requests per minute)
- `L-58BE175A` (cross-region tokens per minute)

Service Quotas > Amazon Bedrock, or the CLI. It is free and does not need paid support.

Until it is approved, leave `LLM_MODEL_ID` unset in the service file. The miss path falls back to the Phase 5 stub, so everything that makes this a cache (embedding, search, the threshold decision, storage, and the hit path) is fully demonstrable without Bedrock at all.
