# Efficient Fine-Tuning Strategies for BERT

### A Systematic Study on SST-2 with PEFT, Data-Centric Learning, and Transfer Learning

**Course Project for AIAA3102, HKUST(GZ)**

**Authors:** Boyi Zhang*, Weifeng Chen*, Yunxiang Li
(* Equal Contribution)

---

## 1. Overview

This project systematically studies **efficient fine-tuning strategies for Transformer-based sentiment classification under realistic GPU and engineering constraints**. Motivated by a startup-style deployment scenario with limited computational resources, we investigate:

* **Full fine-tuning**
* **LoRA fine-tuning (16-bit / 8-bit)**
* **QLoRA fine-tuning (4-bit)**

on the **SST-2 sentiment classification benchmark**, using `bert-base-uncased` as the backbone. Beyond efficiency benchmarking, we further conduct controlled studies on:

* Hyperparameter sensitivity (learning rate, LoRA rank)
* Robustness to randomness and noisy text
* Few-shot generalization and data scaling behavior
* Data-centric difficulty-based sampling
* Model capacity and overfitting via layer freezing
* Cross-task transfer learning from SST-2 to MRPC
* SVD-based stable-rank analysis of LoRA updates
* A deployable **Gradio-based interactive evaluation system**

This repository contains **all training, evaluation, data-centric, transfer-learning, and demo code**, together with **logged experimental results** and **visualization pipelines**.

---

## 2. Reproducibility Policy

We follow standard **reproducible ML experiment practices**:

* All experiments are **configuration-driven** via `src/config.py`
* All metrics are **logged to CSV or JSON**
* All figures are **generated from logged CSV or JSON**
* No manual reporting of numbers
* Each experiment is **deterministic given a random seed using `transformers.set_seed()`**
* Multi-seed results are evaluated using **bootstrap confidence intervals**

The codebase supports **full end-to-end reproduction** of all figures and tables reported in the project report.

---

## 3. Repository Structure

```text
PJ/
├── models/(Models Checkpoints Saved for GUI)                     
│   ├── Full Finetune (FP16)/
│   ├── Full Finetune (FP32)/
│   ├── LoRA (FP16)/
│   ├── LoRA (8-bit)/
│   └── QLoRA (4-bit)/
│
├── notebooks/
│   ├── student_model/(Student Model Checkpoints Saved for Data-Centric Experiments)
│   ├── main.ipynb              
│   └── experiments.ipynb      
│
├── results/
│   ├── csvs/                   
│   ├── figures/               
│   └── transfer learning/     
│
├── src/
│   ├── app.py                  
│   ├── config.py               
│   ├── utils.py                
│   ├── data.py                
│   ├── noisy_data.py           
│   ├── data_centric.py         
│   └── transfer.py             
│
├── requirements.txt
└── README.md
```

---

## 4. Functional Responsibilities

| Module            | Function                                                                                                                      |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `config.py`       | Centralized default configuration of model names, LoRA hyperparameters, dataset variants, random seeds, DeepSeek API key |
| `utils.py`        | Unified pipeline for dataset loading, tokenizer construction, model initialization, training, evaluation, and logging         |
| `noisy_data.py`   | Class-based implementation of character-level and word-level corruption                                                       |
| `data_centric.py` | Difficulty-based data selection and sampling strategies                                                                       |
| `transfer.py`     | Cross-task transfer learning and SVD-based LoRA update analysis                                                               |
| `app.py`          | Gradio-based inference and evaluation interface                                                                               |

---

## 5. Entry-Point Design

### 5.1 `notebooks/main.ipynb` (Instructor / TA Execution Entry)

Designed under the **2-hour Colab runtime constraint**, this notebook provides:

* ✅ Figure generation from pre-logged CSVs
* ✅ Training of **one single model instance (~30 min)**
* ✅ Launching the **Gradio interactive system**

This is the **official execution entry** for evaluation and demonstration.

---

### 5.2 `notebooks/experiments.ipynb` (Full Experimental Suite)

Contains all heavyweight experiments:

* Efficiency benchmarking (Full vs LoRA vs QLoRA)
* Hyperparameter sweeps
* Robustness analysis
* Few-shot scaling
* Data-centric learning
* Capacity control
* Transfer learning
* SVD-based LoRA analysis

This notebook is **not required for real-time demonstration**, but supports **full methodological replication**. Please refer to it for comprehensive experimental details Please notice that you must have at least 300GB of free Google Drive space if you want to run it on Google Colab with mounting to Google Drive.

---

## 6. Supported Fine-Tuning Strategies

| Method         | Backbone Precision | Quantization | Trainable Ratio |
| -------------- | ------------------ | ------------ | --------------- |
| Full Fine-Tune | FP32 / FP16        | None         | ~100%           |
| LoRA (16-bit)  | FP16 / BF16        | None         | ~0.8%           |
| LoRA (8-bit)   | INT8               | 8-bit        | ~0.8%           |
| QLoRA (4-bit)  | INT4               | 4-bit        | ~1.3%           |

All methods use the **same classifier head and tokenization pipeline** to ensure controlled comparison. Results are reported in our report.

---

## 7. Data-Centric Learning Protocol

We construct a **teacher model** using full-data training and compute:

* Per-sample cross-entropy loss
* Difficulty ranking within each label
* Difficulty-stratified subset sampling

We compare:

* Uniform random sampling
* Only easy samples
* Only hard samples
* Middle-only sampling
* **Mixed-difficulty sampling (proposed)**

Please refer to `notebooks/experiments.ipynb` Section 7 for implementation details of Mixed-difficulty sampling.

---

## 8. Transfer Learning Protocol

* **Source Task:** SST-2

* **Target Task:** MRPC

* **Method:**

  1. Train LoRA on SST-2
  2. Transfer LoRA weights to MRPC
  3. Freeze backbone and adapters
  4. Train classifier head only

* Baseline: LoRA trained from scratch on MRPC

* Adapter placements: `attn-qkv`, `attn-full`, `ffn-only`, `full-model`

**LoRA update matrices ΔW = BA are further analyzed using:**

* Per-layer SVD
* Stable-rank statistics
* Cross-task transfer correlation

Implementation:

```
src/transfer.py
```

Running Command Example(after activating conda env):

```bash
cd src
python -u "transfer.py"   --fp16   --num_workers 4   --batch_size 32 --lora_r 16
```

For more details, please refer to the docstrings and comments in `src/transfer.py`.

---

## 9. Gradio Interactive Evaluation System

Implemented in:

```
src/app.py
```

### Supported Features

* Model selection (Full / LoRA / QLoRA / API)
* Single-text prediction
* Softmax confidence reporting
* Real-time latency measurement
* Batch CSV evaluation
* Accuracy & mean latency summary
* Model parameter display

This interface is designed for **non-technical stakeholders** to directly evaluate accuracy–latency–cost trade-offs.

---

## 10. Installation
For setup, create and activate a conda environment with Python>= 3.10 (We've tested on both Windows and Linux, on Python3.10 and Python3.13 on Google Colab), then install dependencies via:
```bash
pip install -r requirements.txt
```

### Dependency List

```text
torch
peft
numpy
scikit-learn
transformers
datasets
accelerate
evaluate
bitsandbytes
gradio
seaborn
pandas
matplotlib
tqdm
tabulate
GPUtil
openai
scipy
```

---

## 11. Execution Protocol (Standardized)

### 11.1 Single-Model Training (Demonstration)

Please refer to `notebooks/main.ipynb` *Section 2* for detailed instructions.

### 11.2 Launch Gradio Interface

```bash
#activate your conda env if needed
cd src
python app.py
```

Access via:

```
http://127.0.0.1:7860(just a reference)
```
if running locally.

Alternatively, use the public share link if running on Colab.

---

## 12. Threats to Validity

* All main experiments are conducted on **SST-2**, a clean and short-text benchmark.
* Only **one cross-task transfer target (MRPC)** is explored.
* Noise modeling is **synthetic character/word-level corruption**.
* LoRA update SVD analysis provides **correlation, not causality**.
* Large-scale LLMs (≥7B) are not tested due to compute limits.

---

## 13. Practical Recommendations

For real-world sentiment classification under limited resources:

* ✅ Use **BERT + LoRA (16-bit)**
* ✅ LoRA rank = 16
* ✅ Learning rate ≈ 1e-4
* ✅ Early stopping within 5–10 epochs
* ✅ ~1000 clean labeled examples are often sufficient
* ✅ Use distribution-preserving data selection
* ✅ Design adapters covering **both attention and FFN**

---

## 14. Team Contribution

| Member           | Contribution                                                                                                                            |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Weifeng Chen** | Project lead; PEFT benchmarking; robustness analysis; capacity control; transfer learning; SVD analysis; report writing; demo recording |
| **Boyi Zhang**   | Data-centric learning; scaling experiments; visualization; presentation slides; report writing                                          |
| **Yunxiang Li**  | Gradio frontend implementation; full experiment reproduction; README documentation                                                      |

## 15. Main Experimental Findings

All findings summarized below are derived from controlled experiments on SST-2 with `bert-base-uncased`, under identical tokenization pipelines, classifier heads, optimization schedules, and training protocols. When applicable, results are averaged over multiple random seeds, with uncertainty estimated using bootstrap confidence intervals.

### 15.1 Efficiency–Performance Trade-off

Across all compared strategies:

+ LoRA (16-bit) consistently achieves comparable or slightly superior accuracy to full fine-tuning, while:
  + Reducing training time to approximately 25–30%
  + Reducing peak GPU memory usage to approximately 50%

+  QLoRA (4-bit) minimizes GPU memory consumption most aggressively, but introduces:

  + Additional quantization and dequantization overhead

  + Longer training time than LoRA (16-bit)

  + LoRA (8-bit) provides limited benefit, as its memory savings are modest relative to QLoRA, while training time remains similar to full fine-tuning.

**Conclusion:**
For medium-scale backbones such as BERT-base, LoRA (16-bit) provides the best overall trade-off among efficiency, stability, and accuracy under practical resource constraints.

## 15.2 Hyperparameter Sensitivity

+ Learning rate:

  + Very small learning rates (≈ 1e-5) lead to underfitting.

  + Performance stabilizes around 1e-4, with robust behavior across nearby values.

+ LoRA rank:

  + Small ranks (e.g., r = 4) underfit.
  + Large ranks (≥ 32) yield diminishing returns and mild overfitting.

  + Rank r = 16 offers the best balance between expressiveness and efficiency.

**Conclusion:**
LoRA exhibits stable performance under moderate hyperparameter tuning, making it suitable for low-budget deployment scenarios where extensive grid search is infeasible.

## 15.3 Robustness to Random Seeds and Input Noise

+ Multi-seed experiments show:

  + Small variance in validation accuracy

  + Highly overlapping 95% bootstrap confidence intervals

  + Noise-augmented training (emoji insertion, character substitution, word deletion) causes less than 1% absolute performance degradation.

**Conclusion:**
The LoRA-based sentiment classifier is robust to both random initialization and realistic user-generated noise.

## 15.4 Scaling Behavior and Few-Shot Learning

+ Zero-shot performance is near random guessing for all methods.

+ Rapid performance growth occurs between 100–1000 training samples.

+ Beyond approximately 1000 samples, validation accuracy quickly saturates.

+ LoRA shows higher variance in extremely low-shot regimes, due to freezing the backbone and optimizing only randomly initialized low-rank adapters.

**Conclusion:**
For SST-2–like clean classification tasks, approximately 1000 well-labeled samples are sufficient to approach full-data performance, and blindly increasing dataset size yields diminishing returns.

## 15.5 Data-Centric Learning

+ Naïve difficulty-based strategies focusing on:

  + Only easy samples

  + Only hard samples

  + Only medium-difficulty samples
  
  all result in degraded performance.

+ A mixed-difficulty sampling strategy, combining predominantly medium-difficulty samples with small proportions of both easy and hard examples, consistently outperforms uniform random sampling.

**Conclusion:**
Preserving the global data distribution is more important than aggressively filtering by perceived difficulty.

## 15.6 Model Capacity and Overfitting

+ Training accuracy continues to rise with epochs, while validation accuracy plateaus early, indicating clear overfitting behavior.

+ Freezing large portions of lower BERT layers:

  + Reduces trainable parameters by up to ~90%

  + While validation accuracy decreases only marginally.

**Conclusion:**
For SST-2, full model capacity is not required, and a substantial portion of BERT’s lower layers is functionally redundant.

## 15.7 Transfer Learning and Adapter Design

+ Transferring SST-2–trained LoRA adapters to MRPC:

  + Significantly accelerates early-stage convergence

  + Outperforms direct LoRA training on MRPC during early epochs
+ **Adapter placement plays a decisive role:**

  + Adapters covering both attention and FFN layers exhibit the strongest transfer performance

  + Extremely narrow adapters (e.g., Q-only or V-only) transfer poorly

  + SVD-based stable-rank analysis shows:
Higher effective rank of LoRA updates correlates with stronger cross-task transferability

**Conclusion:**
LoRA adapters are reusable across related tasks, but successful transfer critically depends on adapter placement and effective representational rank.

## 15.8 Overall Practical Takeaway

For real-world sentiment classification under constrained resources:

✅ Prefer BERT + LoRA (16-bit)

✅ Use LoRA rank ≈ 16 and learning rate ≈ 1e-4

✅ Apply early stopping within 5–10 epochs

✅ Invest in moderate-scale, clean labeled data (~1000 samples)

✅ Use distribution-preserving data selection

✅ Design adapters covering both attention and FFN layers to support future task transfer