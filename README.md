# 📘 Sentiment Analysis with a Custom Transformer
IMDB Movie Reviews — PyTorch Implementation

## 📝 Executive Summary

This project implements a Transformer-based sentiment analysis model entirely from scratch, trained on the IMDB movie reviews dataset.
Instead of relying on pretrained models like BERT, every component (attention heads, embeddings, transformer blocks, and classifier) was manually coded using PyTorch.

The final model achieves:

- Validation Accuracy: 79.36%

- Test Accuracy: 76.33%

While not as high as pretrained models, this result is strong for a Transformer trained on limited data, and the project’s main value lies in the deep technical understanding developed.

----

## 📌 Project Summary

In this project, I built and trained a custom Transformer architecture for sentiment classification.
The implementation includes:

### 🔧 Model Architecture

- Token embeddings
- Positional embeddings
- Multi-head self-attention
- Feed-forward networks
- Residual connections
- Layer normalization
- Dropout regularization
- Classification head for binary sentiment prediction

### 📚 Data Pipeline

- Custom tokenizer-based preprocessing
- IMDB dataset loading
- Train/test splits
- PyTorch DataLoaders
- Padding and sequence management

### 🧠 Training

- AdamW optimizer
- CrossEntropyLoss
- Gradient clipping
- Validation after each epoch
- Final evaluation on the test set

----

## 📊 Results

Overall Performance

| Metric                  | Value      |
| ----------------------- | ---------- |
| **Validation Accuracy** | **79.36%** |
| **Test Accuracy**       | **76.33%** |

**Key Observations**

- Training loss converged after ~3 epochs
- Most misclassifications occur on borderline reviews (mixed sentiment)
- Probability distributions show clear separation between positive and negative predictions
- The ROC curve confirms the model has good discriminative ability despite being trained

----

## 📉 Visual Analysis

Included in the notebook:

1. Learning Curves

- Training Loss steadily decreases
- Validation accuracy increases consistently

2. Confusion Matrix

- Shows more false negatives than false positives
- Borderline/mixed sentiment is the main challenge

3. ROC Curve

- Smooth, well-shaped AUC behavior

4. Probability Distribution Plots

- Clear separation between predicted sentiment classes

----

## 🎯 Key Takeaways

🔹 1. Deep Understanding of Transformer Mechanics

By coding every component manually, I developed intuition for:

- attention score computation
- masking
- multi-head architecture
- residual stability
- normalization impact

This knowledge applies directly to modern LLM architectures.

🔹 2. Building a Model from Scratch

I independently created and validated each module:

- AttentionHead
- MultiHeadAttention
- FeedForward
- Block
- DemoGPT-style full Transformer

This improves debugging skills and architectural literacy.

🔹 3. Data Processing Matters

Tokenization, padding, batching, and DataLoaders are as crucial as the model itself.

🔹 4. Strong ML Engineering Practices

- Proper training/eval split
- Validation monitoring
- Test-only final evaluation
- Logging metrics
- Saving reproducible checkpoints

🔹 5. Complete End-to-End NLP Pipeline

From raw dataset → preprocessing → training → evaluation → visualization.

----

## 📂 Repository Structure

```bash
📁 Sentiment-Analysis-using-PyTorch/
│ 
├── SentimentScope_starter.ipynb                   # Notebook transformer model 
├── model_checkpoint.pt                            # Checkpoint
├── executive.md                                   # Executive Summary
├── /plots                                         # Visualizations and charts 
│     ├── Compare Probability Distributions.png
│     ├── Probability Distribution.png
│     ├── probability_histogram.png
│     ├── Confusion matrix.png
│     ├── Label Distribution in Training Set.png
│     ├── Review Length by Sentiment (Characters).png
│     ├── Review Length by Sentiment (Words).png
│     ├── Review Length Distribution (Characters).png
│     ├── Review Length Distribution (Words).png
│     └── ROC Curve.png
│ 
├── README.md                                     # Project overview
└── requirements.txt                              # Libraries required to run the project
```
----

## 📥 Dataset Download

The IMDB movie reviews dataset is not included in this repository because of its size.
You must download it manually from Stanford:

🔗 Download IMDB Dataset (ACL IMDB v1)
https://ai.stanford.edu/~amaas/data/sentiment/

After downloading:

1. Extract the .tar.gz file.

2. You will obtain the folder:
   
```bash
aclImdb/
    ├── train
    ├── test
    ├── imdb.vocab
    └── README
```

3. Place it in the project directory:

4. The notebook automatically loads the dataset from this folder.

----

## 🧪 How to Run

Follow these steps to fully reproduce the Transformer model training and evaluation:

1. Set up the environment

Create a clean Python environment:

```bash
python -m venv .venv
source .venv/bin/activate     # macOS / Linux
.venv\Scripts\activate        # Windows
```

Install all required packages:

```bash
pip install -r requirements.txt
```

2. Download the Dataset (Required)

The IMDB sentiment dataset must be downloaded manually:

🔗 https://ai.stanford.edu/~amaas/data/sentiment/

Extract it and ensure the project structure looks like this:

```bash
/project_root
    ├── aclImdb/
    ├── notebook.ipynb
    ├── README.md
    ├── requirements.txt
```

3. Run the Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open notebook.ipynb, then:

- Run preprocessing cells
- Train the Transformer model
- Generate visualizations
- Evaluate test accuracy

The notebook will also save a model checkpoint automatically.

----

## 🔮 Future Work

- Increase model embedding size and number of heads
- Train for more epochs with regularization tuning
- Experiment with pretraining (Masked Language Modeling)
- Compare against pretrained baselines (BERT, DistilBERT)
- Deploy as an API or Streamlit app

----

## Author
**Bárbara Ángeles Ortiz**

<img src="https://github.com/user-attachments/assets/30ea0d40-a7a9-4b19-a835-c474b5cc50fb" width="115">

[LinkedIn](https://www.linkedin.com/in/barbaraangelesortiz/) | [GitHub](https://github.com/BarbaraAngelesOrtiz)

![Status](https://img.shields.io/badge/status-finished-brightgreen) 📅 November 2025

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-orange)
![TorchText](https://img.shields.io/badge/TorchText-red)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-green)
![Pandas](https://img.shields.io/badge/Pandas-purple)
![NumPy](https://img.shields.io/badge/NumPy-lightblue)

![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Colab](https://img.shields.io/badge/Google%20Colab-Notebook-yellow)

----

## Acknowledgments

<img width="205" height="205" alt="aws-logo-icon (1)" src="https://github.com/user-attachments/assets/96be7e16-e0db-43c8-bdc4-83569ce3eb1c" />

<img width="205" height="205" alt="images (2)" src="https://github.com/user-attachments/assets/7084389d-848e-4802-a85a-4763cef95d0c" />

