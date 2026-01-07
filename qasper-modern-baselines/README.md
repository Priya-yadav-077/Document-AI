# LLM Baselines for Qasper Dataset

This folder contains modern baseline implementations for the Qasper dataset, following the same evaluation protocol as the original LED baseline (2021).

## Setup

### 1. Install dependencies
```bash
cd qasper-modern-baselines
pip install -r requirements.txt
```

### 2. Download Qasper dataset
```bash
python3 scripts/download_qasper.py
```

### 3. Set up API keys
Create `.env` file in this directory:
```
GOOGLE_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
```

## 4. (Possible) Evaluation for Baseline Creation

Here's the complete workflow from start to finish:

```bash
# 1. Quick model comparison
python3 scripts/run_evaluation.py --split dev --num_samples 1 --model gemini-2.5-flash
python3 scripts/run_evaluation.py --split dev --num_samples 1 --model gemini-2.5-pro
python3 scripts/run_evaluation.py --split dev --num_samples 1 --model llama-3.3-70b

# 2. Dev set validation with best models (60 minutes)
python3 scripts/run_evaluation.py --split dev --num_samples 50 --model gemini-2.5-pro
python3 scripts/run_evaluation.py --split dev --num_samples 50 --model llama-3.3-70b

# 3. Final test set evaluation (3-4 hours)
python3 scripts/run_evaluation.py --split test --num_samples 50 --model gemini-2.5-pro
python3 scripts/run_evaluation.py --split test --num_samples 50 --model llama-3.3-70b

```
