# LLM Inference Infra (Gemma 2 Inference Server + Terraform/Ansible)

This project provides a **production-lean inference stack** for running a Gemma‑2 (2B/27B) model with optional **LoRA** adapters behind a lightweight **Flask** HTTP API, plus **Terraform** modules to provision an EC2 host (VPC, subnet, security group, EBS) and an **Ansible** inventory to operate the host.

The focus is on **clear, reproducible setup**: local dev, single‑VM deploy, and a path to scale. 

GPU infrastructure is hosted at UCSB. Intermediate infra (AWS lambda, sms integrations, whats app integration, we chat integration ) is omitted from this repo. The demo shows the real world usage of inference server.

<div style="display: flex; gap: 10px;">
  <img src="media/sms.gif" width="49%" alt="SMS Demo">
  <img src="media/wechat.gif" width="49%" alt="Wechat Demo">
</div>


---

## 1) Features (What this is)
- **Flask JSON API** (`server.py`) that wraps Gemma‑2 models (`google/gemma-2-2b-it` or `google/gemma-2-27b-it`) with a consistent prompt for **carbohydrate estimation**.
- **LoRA adapter** support for the 27B model (`peft.PeftModel`).
- **Cleaned responses**: server extracts only the model’s `<start_of_turn>model ... <end_of_turn>` segment.
- **Logging** to file and console (`logs.txt`).
- **Infra as Code**: Terraform to spin up an EC2 with VPC, subnet, routes, SG, EBS; Ansible inventory for remote ops.
- **Standalone CLI inference** utilities (`inference.py`, `inference_gemma.py`, `inference1.py`) for quick tests and batch runs.

---

## 2) Repo Layout
```
llmserver/
├── server.py                 # Flask JSON API
├── inference.py              # Class-based Gemma 2B/27B inference (LoRA optional)
├── inference_gemma.py        # Pipeline-based inference w/ JSON extraction helper
├── inference1.py             # Alt pipeline-based inference
└── deploy/
    └── terraform/
        ├── main.tf          # EC2 + EBS + user_data mount + EIP
        ├── network.tf       # VPC, subnet, IGW, route table, SG
        ├── variables.tf     # AMI/instance/keys vars
        └── volume.tf        # EBS volume + attachment
    └── ansible/
        └── inventory.ini    # Host inventory (edit IP/SSH key)
```

---

## 3) Local Development (Quickstart)
### 3.1 Environment
```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate
pip install -U torch transformers peft flask
```

> For GPU acceleration and bf16, install the CUDA‑matched PyTorch build from pytorch.org.

### 3.2 Run the server (2B or 27B)
```bash
# 2B IT model from HF Hub
python server.py --model_type 2b --port 5000

# 27B IT model with an optional LoRA adapter
python server.py --model_type 27b --adapter_path /path/to/lora_dir --port 5000
# (omit --adapter_path if you don't use LoRA)
```

### 3.3 Test the API
```bash
curl -X POST http://localhost:5000/   -H "Content-Type: application/json"   -d '{"prompt": "Half a peanut butter and jelly sandwich."}'
# → {"response": "{\"total_carbohydrates\": 25.3}"}
```

**Request body**
```json
{ "prompt": "<meal description text>" }
```
**Response body**
```json
{ "response": "<model_text_cleaned>" }
```

### 3.4 CLI Inference Utilities
- `inference.py` (class-based, close to server behavior):
```bash
python inference.py --model_type 2b --prompt "1 cup oatmeal and a banana"
```
- `inference_gemma.py` (pipeline + JSON extraction):
```bash
python inference_gemma.py --model_path /path/to/model --prompt "PB&J sandwich"
```
- `inference1.py` (alternate pipeline runner):
```bash
python inference1.py --model_path /path/to/model --prompt "Orange juice 8oz"
```

---

## 4) Data & Model Notes
- **Models**: pulls from HF Hub by default. Use `--local_model_dir` to point to a local checkout/cache.
- **bf16**: defaults are set to `torch.bfloat16` on capable GPUs.
- **LoRA**: for 27B, pass `--adapter_path` to apply `peft` adapters.
- **Prompting**: the server uses a **CoT‑style Gemma prompt** with few‑shots and expects a final **JSON** answer containing `total_carbohydrates`.

---

## 5) AWS Deployment (Terraform single‑VM)
> Files: `deploy/terraform/*.tf`

### 5.1 Prereqs
- Terraform ≥ 1.2, AWS credentials configured, and an existing **key pair** in the chosen region.
- Edit `variables.tf` defaults as needed:
  - `ami_id` (Ubuntu/AL2 in `us-west-2` by default)
  - `instance_type` (start with `t2.medium` for CPU tests; use a GPU instance for production)
  - `aws_key_pair_name`, `local_private_key_path`

### 5.2 Provision
```bash
cd deploy/terraform
terraform init
terraform apply -auto-approve
```
Outputs will include **SSH instructions** and the **Elastic IP**.

> The instance user data formats & mounts an EBS volume at `/mnt/data` and persists it via `/etc/fstab`.

### 5.3 Security (Important)
- `network.tf` opens SSH (22), HTTP (80), HTTPS (443), and Flask dev port (5000) to **0.0.0.0/0** for convenience.  
  **Harden these rules** before internet exposure:
  - Limit source CIDRs (e.g., your office/home IPs).
  - Prefer a proper reverse proxy (Nginx) on 443 and keep Flask behind the VPC.
  - Consider AWS ALB + private subnets.

---

## 6) Remote Operation (Ansible)
> File: `deploy/ansible/inventory.ini`

1) Replace the IP and key path with your EC2’s values.  
2) Example ad‑hoc check:
```bash
ansible -i deploy/ansible/inventory.ini host -m ping
```
3) You can then author a playbook to:
   - install Python deps,
   - create a systemd unit for `server.py`,
   - set up Nginx as a TLS terminator,
   - enable log rotation for `logs.txt`.

---

## 7) Productionizing Tips
- **Process manager**: run Flask behind `gunicorn` or `uvicorn` + `systemd`.
- **Reverse proxy**: terminate TLS with Nginx/Caddy; route `/` → Flask on localhost.
- **Autoscaling**: bake an AMI with the model weights; use an **Auto Scaling Group** behind an **ALB**.
- **Caching**: enable `HF_HOME` on the EBS volume to reuse weights; warm up at boot.
- **Observability**: ship logs to CloudWatch; export latency/throughput to Prometheus/Grafana.
- **Throughput**: batch requests; consider vLLM/text‑generation‑inference for high‑QPS serving.
- **Limits**: validate request sizes; set generation timeouts; protect the endpoint with auth.

---

## 8) Repro Checklist
1. **Local**: create venv, install deps, run `server.py` with `--model_type 2b`, send a `curl` request.  
2. **Cloud**: `terraform apply`, SSH into the instance, install Python deps, run `server.py` on port 5000, test via public IP.  
3. **Harden**: restrict SG, add Nginx TLS, daemonize the app, set up log rotation.  
4. **Optional**: attach LoRA adapters for 27B, point to local model dir to reduce cold‑start.

---

## 9) API Contract (Reference)
**POST /**  
Request:
```json
{ "prompt": "Half a PB&J sandwich" }
```
Response:
```json
{ "response": "{"total_carbohydrates": 25.3}" }
```

---

## 10) Troubleshooting
- **CUDA/Driver mismatch**: install CUDA‑matched PyTorch; verify `torch.cuda.is_available()`.
- **Out of memory**: try 2B model first; reduce `max_new_tokens`; prefer A10/A100 class GPUs.
- **Tokenizer pad/eos**: Gemma tokenizer sets both; if missing, the code assigns fallbacks.
- **HF auth**: if the model is gated, run `huggingface-cli login` on the server.

---

## 11) License & Credits
- Uses Hugging Face **Transformers**, **PEFT**, **Flask**, **Terraform**.
- Models: Google **Gemma‑2** family.
