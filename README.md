# Depression Detection from Blog Posts

This project explores automatic detection of depressive language in online blog posts using modern NLP methods. The task is framed as a **binary text classification problem**, comparing a fine-tuned transformer model with a zero-shot prompting baseline.

---

## Problem Description

This project investigates whether modern NLP models can distinguish **depressive language** from other forms of mental health–related expression in online blog posts.

The original dataset contains multiple mental health categories (e.g., depression, anxiety, PTSD/trauma, suicidal thoughts and self-harm). To focus the task and enable clear evaluation, the problem is simplified into a **binary classification setting**:

- **Depression** → posts explicitly or implicitly expressing depressive symptoms  
- **Non-depression** → posts related to other mental health concerns (e.g., anxiety, trauma, self-harm) that may contain emotionally intense language but do not primarily indicate depression

This framing tests whether models can capture **subtle linguistic and contextual differences** between depression and other emotionally intense but distinct mental health conditions, rather than simply detecting negative sentiment. The task therefore emphasizes nuanced interpretation of long-form, self-expressive text.

---

## Dataset

This project uses the **Mental Health Blog (MHB) Dataset**, introduced by Boinepelli et al. (2022). The dataset consists of long-form user-authored posts collected from online mental health forums and annotated with several mental health–related categories, including depression, anxiety, PTSD/trauma, and suicidal thoughts and self-harm.

For this project, the dataset is adapted to the binary classification setting described above, enabling direct evaluation of a model’s ability to distinguish depression-specific linguistic patterns from broader mental health discourse.

**Source:**  
Boinepelli et al., *Leveraging Mental Health Forums for User-Level Depression Detection on Social Media*, LREC 2022.

---

## Methods

### 1. Fine-Tuned RoBERTa
- Model: `roberta-base`
- Supervised fine-tuning on labeled training data
- Evaluation on a held-out test set
- Metrics: accuracy, precision, recall, macro-F1

### 2. Zero-Shot Prompting Baseline
- Model: `facebook/bart-large-mnli`
- No task-specific training
- NLI-based zero-shot classification using label entailment
- Evaluated on a random subset of 200 test examples for efficiency

---

## Results Summary

- The fine-tuned RoBERTa model achieved a macro-F1 score of approximately **0.92** on the test set.
- The zero-shot prompting baseline achieved a macro-F1 score of approximately **0.66** on a random subset of 200 test examples.

The supervised model substantially outperformed the prompting baseline, highlighting the importance of task-specific fine-tuning for nuanced mental health text classification.
