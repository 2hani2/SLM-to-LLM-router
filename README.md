# NanoQA: SLM-to-LLM Router

## Project Overview

This project implements a **confidence-gated SLM–LLM routing system** for efficient question answering.

Instead of using a single large model for all queries, the system dynamically routes each query to the most appropriate component, improving latency, cost, and efficiency.

---

## System Architecture

The system follows a three-stage routing pipeline:

1. Symbolic Math Engine
   Handles mathematical expressions instantly (~1 ms)

2. NanoQA (Small Language Model - 135M parameters)
   Handles factual, short-answer queries
   Trained from scratch using curated datasets

3. Mistral 7B (Large Language Model)
   Handles complex queries requiring reasoning

---

## Core Idea

Instead of training a separate classifier, routing is based on model confidence:

* The system computes the mean softmax probability of generated tokens
* If confidence ≥ 0.60 → accept SLM output
* If confidence < threshold → escalate to LLM

This eliminates the need for:

* Labeled routing data
* External classifiers
* Additional training overhead

---

## Methodology

* Dataset creation and augmentation (300k+ QA pairs)
* Token-level training using:

  * Focal Loss (γ = 2)
  * Knowledge Distillation (from GPT-2)
* Confidence-based routing using softmax probabilities
* Evaluation using:

  * Accuracy, ROUGE, F1, MRR
  * Routing Precision, Recall, F1
  * Latency measurements

---

## Results

According to evaluation:

* 98.0% accuracy
* 98.6% MRR
* 82.1% routing F1
* 63% reduction in total response time
* ~13× speedup compared to always using LLM

---

## Project Structure

```bash
.
├── router.py                  # Core routing logic
├── model.py                   # NanoQA model
├── train.py                   # Training pipeline
├── tokenizer_train.py         # Tokenizer
├── generate_training_data.py  # Data generation
├── collect_and_visualize.py   # Metrics collection
├── dynamic_metrics_dashboard.py
├── new_visualizations.py
├── app.py                     # Main application
├── new_handcrafted_qa.json    # Dataset
├── handcrafted_qa.json        # Dataset
```

---

## Dataset

The dataset is included in this repository:

* Handcrafted QA pairs
* Augmented dataset for training
* Domain-specific curated data

---

## How to Run

```bash
git clone https://github.com/2hani2/SLM-to-LLM-router.git
cd SLM-to-LLM-router
pip install -r requirements.txt
python train.py
python app.py
```

---

## Key Contributions

* Confidence-based routing without a trained classifier
* Fully local SLM–LLM hybrid system
* Efficient small model outperforming larger models on domain tasks
* Significant reduction in latency and compute cost

---

## Future Work

* Improve paraphrase understanding using embeddings
* Expand NanoQA to larger parameter sizes
* Integrate reinforcement learning for routing

---

## Author

Venisa
Manipal Institute of Technology
