# CS550 — Massive Data Mining & Learning

## Course Project: Graph Analysis on the Cora Citation Network

![CI](https://github.com/lk671/cs550-cora-graph-project/actions/workflows/ci.yml/badge.svg)

---

## 📌 Project Overview

This project implements **graph-based machine learning** on the [Cora citation network dataset](https://relational.fit.cvut.cz/dataset/CORA) as part of **CS550 Project Option 2: Social Networks**.

The goal is to:

1. **Node Classification** — Predict paper topics based on content and citations (GCN/GAT)
2. **Link Prediction** — Predict missing citations using Variational Graph Auto-Encoders (VGAE)
3. **Interactive Dashboard** — A 3D-graph explorer with real-time inference
4. **Trustworthiness & Ethics** — Integrated xAI, Fairness Audits, and Robustness stress-testing

**Dataset:** Cora — 2,708 papers, 10,556 citation edges, 7 topic classes, 1,433 word features per paper.

---

## 👥 Contributors

- Mayank Shetty Sreekanth
- Lokesh Kota
- Prajwal Srinivas

---

## ⚙️ Setup Instructions

> Do this **once** when you first clone the repo.

### 1. Clone the Repository

```bash
git clone https://github.com/lk671/cs550-cora-graph-project.git
cd cs550-cora-graph-project
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment

```bash
# Mac / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

> ✅ You'll know it's active when you see `(venv)` at the start of your terminal prompt.
> Run this **every time** you open a new terminal to work on the project.

### 4. Install All Dependencies

```bash
pip install -r requirements.txt
```

> This installs the exact same library versions as everyone else on the team.
> No version mismatch issues.

---

## ▶️ Running the Project

### Verify your setup works (run these first):

```bash
# Downloads Cora dataset and prints graph statistics
python data_loader.py

# Creates 80/20 split and runs Logistic Regression baseline
python baseline_model.py
```

### Train link prediction models

```bash
# Train GAE and VGAE on Cora link prediction
python train_link.py
```

What this does:

- Uses `RandomLinkSplit` to create train/validation/test edge splits
- Trains a GAE baseline and a VGAE main model
- Tunes the best decision threshold on validation F1
- Saves model weights to `models/gae_best*.pth` and `models/vgae_best*.pth`
- Writes link metrics and ROC artifacts into `outputs/`

Useful outputs:

- `outputs/results*.csv`
- `outputs/gae_roc_data*.npz`
- `outputs/vgae_roc_data*.npz`

### Generate report plots

```bash
# Create the comparison charts and ROC figure
python plot.py
```

What this does:

- Builds the node classification comparison chart for LR, GCN, and GAT
- Builds the link prediction comparison chart for GAE and VGAE
- Builds the grouped link metrics table/heatmap for the report
- Builds the ROC curve for link prediction

Useful outputs:

- `outputs/node_classification_comparison.png`
- `outputs/gae_vgae_comparison.png`
- `outputs/gae_vgae_table.png`
- `outputs/link_prediction_roc.png`

### Run the full pipeline (once all modules are complete):

```bash
python main.py
```

> Outputs metrics to `outputs/results.csv` and charts to `outputs/`

### Run the interactive demo:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
cs550-cora-graph-project/
│
├── .github/workflows/
│   └── ci.yml              # CI/CD — auto-tests code on every push
│
├── data/                   # Cora dataset (auto-downloaded, NOT committed to Git)
├── models/                 # Saved model weights (.pth files, NOT committed to Git)
├── outputs/                # Generated charts and results.csv (NOT committed to Git)
│
├── data_loader.py          # Downloads Cora, prints dataset statistics
├── baseline_model.py       # 80/20 stratified split + Logistic Regression baseline
├── metrics.py              # Precision, Recall, F-measure, NDCG calculations
├── models.py               # GCN / GAT model architecture definitions
├── train_node.py           # Node classification training loop
├── train_link.py           # Link prediction training loop (GAE/VGAE)
├── plot.py                 # Generates comparison charts for the report
├── main.py                 # Master pipeline — runs everything end to end
├── app.py                  # Streamlit interactive demo
│
├── robustness.py          # Adversarial stress-testing engine
├── explainer.py           # xAI logic (GNNExplainer + neighborhood analysis)
├── llm_explain.py         # Natural language insights via Groq/OpenAI APIs
├── requirements.txt        # Locked library versions
└── README.md               # You are here
```

---

## 🌿 Git Branching Strategy

**Never push directly to `main`.** Use feature branches:

```bash
# Create your branch when starting a new task
git checkout -b feature/your-task-name

# Example branch names:
# feature/gcn-node-classification
# feature/gae-link-prediction
# feature/streamlit-demo
# feature/metrics-engine

# Push your branch to GitHub
git push origin feature/your-task-name
```

Then open a **Pull Request** on GitHub → get it reviewed → merge into `main`.

---

## 🔄 CI/CD Pipeline

Every push triggers an automated test on GitHub Actions:

- Installs all dependencies on a clean Ubuntu machine
- Runs `data_loader.py` and `baseline_model.py`
- A **green ✅** next to a commit means the code is healthy
- A **red ❌** means something broke — check the Actions tab for details

---

## 📊 Final Performance Metrics

| Model                          | Feature/Metric | Score / Accuracy | Status |
| ------------------------------ | -------------- | ---------------- | ------ |
| Logistic Regression (Baseline) | Accuracy       | 76.57%           | ✅ Baseline |
| GCN (Graph Convolution)        | Accuracy       | **88.93%**      | ✅ Complete |
| GAT (Graph Attention)          | Accuracy       | 88.19%           | ✅ Complete |
| VGAE (Link Prediction)         | AUC-ROC        | **0.934**        | ✅ Complete |

---

## ✅ Final Submission Checklist

Before zipping the project, verify that:
- [ ] **Data exists**: `data/Cora/raw/` contains the 8 Planetoid files.
- [ ] **Models trained**: `models/` contains `gat_best.pth` and `vgae_best.pth`.
- [ ] **Outputs populated**: `outputs/` contains `results.csv` and necessary PNG plots.
- [ ] **Environment ready**: `pip install -r requirements.txt` has been run.
- [ ] **Core files**: `app.py`, `explainer.py`, `robustness.py`, and `llm_explain.py` all present.

---

## 📦 Key Dependencies

| Library                  | Version | Purpose                      |
| ------------------------ | ------- | ---------------------------- |
| `torch`                  | 2.8.0   | Deep learning engine         |
| `torch-geometric`        | 2.6.1   | Graph neural network toolkit |
| `scikit-learn`           | 1.6.1   | Baseline models & metrics    |
| `matplotlib` / `seaborn` | —       | Charts for the report        |
| `streamlit`              | TBD     | Interactive demo web app     |
