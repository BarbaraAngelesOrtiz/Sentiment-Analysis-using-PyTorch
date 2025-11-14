Executive Summary – SentimentScope: Transformer-Based Sentiment Analysis for Cinescope

As a Machine Learning Engineer at Cinescope, I developed SentimentScope, a transformer-based sentiment analysis model trained from scratch using the IMDB movie reviews dataset. This system enhances Cinescope’s recommendation engine by identifying user sentiment in written reviews, enabling more personalized content experiences.

This project demonstrates end-to-end capability in dataset processing, model implementation, training, evaluation, and reporting.

1. Project Objectives

✔ Train a custom transformer model from scratch using PyTorch

✔ Perform binary sentiment classification on IMDB reviews

✔ Achieve over 75% test accuracy

✔ Generate a complete project report and visual results

✔ Demonstrate mastery of attention mechanisms, embeddings, and transformer architecture

2. Technical Approach
Dataset

IMDB Reviews — 50,000 samples

Balanced classes: 25k positive / 25k negative

Train / Validation / Test split

Preprocessing

Tokenization with HuggingFace tokenizer

128-token sequence length

Attention mask generation

Custom PyTorch Dataset + DataLoader

Model Architecture

A compact transformer classifier built entirely from scratch:

Token + position embeddings

1 transformer block

Multi-head attention (4 heads)

Feed-forward network

Dropout regularization

Classification head (2 classes)

Despite being small, the model converged in 3 epochs and achieved solid performance.

Training Setup

Loss: CrossEntropyLoss

Optimizer: Adam

Epochs: 3

Batch Size: 32

Device: CPU

3. Results
3.1 Training & Validation Performance
Epoch	Validation Accuracy
1	68.88%
2	76.28%
3	78.36%

Model shows clear convergence and improving validation accuracy.

3.2 Test Performance (Final Evaluation)
Metric	Value
Test Accuracy	76.92%
AUC (ROC)	0.854
F1 Score	0.77
Precision	0.78
Recall	0.76
✔ Meets requirement: >75% test accuracy
3.3 Confusion Matrix

(rows = true labels, columns = predicted labels)

	Predicted Negative	Predicted Positive
True Negative	9838	2662
True Positive	3108	9392

The model performs reasonably well on both classes.

Most errors come from false positives/negatives on borderline reviews.

3.4 Model Behavior Observations

Loss decreases steadily across epochs.

No signs of overfitting after 3 epochs.

ROC AUC of 0.854 indicates good separation between classes.

4. Key Takeaways
1. Building a Transformer from Scratch is Effective for Sentiment Tasks

Even without pretraining, a compact transformer can achieve 76–78% accuracy, proving that attention-based architectures learn sentiment cues efficiently.

2. IMDB Sentiment Is Complex — Errors Come from Ambiguous Reviews

Analysis of false positives/negatives shows:

Irony

Mixed sentiment

Long reviews with contradictory statements

These require deeper contextual understanding — something larger, pretrained models would handle better.

5. Future Improvements

To surpass 90% accuracy:

Fine-tune pretrained models such as DistilBERT, BERT-base, RoBERTa

Increase model depth (more transformer blocks)

Use longer sequences (256–512 tokens)

Add interpretability tools (attention visualizations, SHAP)

Deploy an inference API (FastAPI)

6. Repository Structure
SentimentScope/
│
├── data/                  # Dataset (preprocessed)
├── src/                   # Transformer model, training/inference scripts
├── notebooks/             # Full training and evaluation notebook
├── outputs/               # Confusion matrix, ROC curve, plots
├── executive.md           # Executive summary (this file)
└── README.md              # Setup & run instructions

7. Conclusion

SentimentScope successfully meets all industry best-practice requirements:

Achieves 76.92% accuracy on the test set (✔ requirement > 75%)

Includes a complete evaluation report

Provides meaningful insights and visual analysis

Demonstrates the ability to design and train a transformer from scratch

This project strengthens Cinescope’s ability to analyze user feedback and power more personalized recommendation strategies.
