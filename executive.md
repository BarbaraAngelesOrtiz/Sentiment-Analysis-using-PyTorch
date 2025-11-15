# Executive Summary – SentimentScope: Transformer-Based Sentiment Analysis for Cinescope

As a Machine Learning Engineer, I developed a transformer-based sentiment analysis system trained from scratch using the IMDB movie reviews dataset. This model supports Cinescope’s recommendation engine by identifying sentiment in user reviews, enabling more personalized content experiences and improved audience understanding.

This project demonstrates full-stack ML capability: data ingestion, preprocessing, custom transformer implementation, model training, evaluation, interpretability analysis, probability diagnostics, and reporting.

----

## 1. Project Objectives

✔ Train a custom transformer model from scratch using PyTorch

✔ Perform binary sentiment classification on IMDB reviews

✔ Achieve over 76.33% test accuracy

✔ Generate a complete project report with visual results

✔ Demonstrate mastery of attention mechanisms, embeddings, and transformer architecture

----

## 2. Technical Approach

**Dataset**

- IMDB Reviews — 50,000 samples

- Balanced classes: 25k positive / 25k negative

- Train / Validation / Test split

**Preprocessing**

- Tokenization with HuggingFace tokenizer

- 128-token sequence length

- Attention mask generation

- Custom PyTorch Dataset + DataLoader
  
- Stratified train/val/test split

**Model Architecture**

A compact transformer classifier built entirely from scratch:

- Token + positional embeddings

- 1 transformer block

- Multi-head attention (4 heads)

- Feed-forward network

- Dropout regularization

- Final dense classification head

The model converged in 3 epochs, showing stable and consistent learning.

**Training Setup**

- Loss: CrossEntropyLoss

- Optimizer: Adam

- Epochs: 3

- Batch Size: 32

- Device: CPU
  
----

## 3. Results
### 3.1 Training & Validation Performance

| Epoch | Validation Accuracy |
| ----- | ------------------- |
| 1     | 70.40%              |
| 2     | 75.88%              |
| 3     | 79.36%              |

The model shows clear convergence and stable improvement.

###  3.2 Final Test Performance

| Metric            | Value      |
| ----------------- | ---------- |
| **Test Accuracy** | **76.33%** |
| **AUC (ROC)**     | **0.847**  |

✔ Exceeds requirement: >75% accuracy

✔ Strong AUC (0.85) indicates excellent class separability

<img width="613" height="624" alt="ROC Curve" src="https://github.com/user-attachments/assets/bf3978be-67de-41bb-ba83-4d1e3af8fa28" />

### 3.3 Confusion Matrix

|                   | Predicted Negative | Predicted Positive |
| ----------------- | ------------------ | ------------------ |
| **True Negative** | 8837               | 3663               |
| **True Positive** | 2255               | 10245              |

- The model is slightly stronger at identifying positive reviews.
- Most misclassifications fall in borderline or mixed-sentiment cases.
- Error rates align with typical behavior of compact transformers trained from scratch.

### 3.4 Probability Distribution Analysis 

Two complementary probability analyses were performed.

<img width="704" height="470" alt="Probability Distribution" src="https://github.com/user-attachments/assets/08ea9d6f-5399-4fc0-8190-356f548d5cd2" />

#### 1. Class-Separated Probability Histogram

Key insights:

- Negative reviews cluster near 0.0–0.1 → high confidence for negative sentiment.
- Positive reviews cluster near 0.9–1.0 → high confidence for positive sentiment.
- Overlap appears in the 0.4–0.6 region → where mixed-tone, ironic, or neutral reviews reside.
- Misclassifications occur mostly in this overlapping band.

This indicates strong calibration at the extremes and uncertainty only in truly ambiguous samples.

<img width="704" height="470" alt="Compare Probability Distributions" src="https://github.com/user-attachments/assets/020939bc-a416-476c-9b02-68c5e04be2f4" />

#### 2. Global Probability Distribution (U-shaped curve)

- Sharp peaks at 0.0 and 1.0.
- Very few samples around 0.5.
- Confirms that the model avoids uncertain predictions and is confident when the sentiment signal is strong.

Combined, both analyses show:

- Robust polarity detection
- Clear confidence patterns
- Good interpretability and model calibration

----

## 4. Key Takeaways
### 1. Transformer from Scratch = Effective

Even without pretraining, a compact transformer can reach 76–78% accuracy, demonstrating that attention mechanisms capture sentiment cues well.

### 2. Probability Distributions Reveal Strong Decisiveness

The U-shaped distribution and class-separated histograms show:

- High confidence in clear cases
- Ambiguity only in mixed-sentiment reviews
- Good alignment between predicted confidence and human interpretability

### 3. Complex Sentiment Requires Larger Models

Sarcasm, mixed emotions, and subtle negativity are best handled by pretrained models like BERT, DistilBERT, or RoBERTa.

----

## 5. Future Improvements

To push the model beyond 90% accuracy:

- Fine-tune a pretrained transformer (DistilBERT, BERT-base, RoBERTa)
- Increase depth (more transformer blocks)
- Use longer input sequences (256–512 tokens)
- Add interpretability tools (attention maps, SHAP values)
- Deploy an inference API using FastAPI

----

## 6. Repository Structure

```bash

SentimentScope/
│
├── data/                  # Dataset (preprocessed)
├── src/                   # Transformer model, training/inference scripts
├── notebooks/             # Full training and evaluation notebook
├── outputs/               # Confusion matrix, ROC curve, probability plots
├── executive.md           # Executive summary (this file)
└── README.md              # Setup & run instructions
```
----

## 7. Conclusion

SentimentScope successfully meets all project requirements:

✔ 76.33% test accuracy

✔ Strong AUC of 0.847

✔ High-confidence probability behavior

✔ Full end-to-end implementation of a custom transformer

✔ Meaningful insights for recommendation systems


