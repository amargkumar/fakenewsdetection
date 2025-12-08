# fakenewsdetection
Goal is to design, train, and evaluate machine learning models that detect fake vs. real health information

A Machine Learning Architecture for Detecting Fake Health Information

## Overview

This project implements and compares multiple neural network architectures for detecting health misinformation across two datasets (CoAID and HealthFact). The system evaluates:

- **Feed-Forward Neural Networks (FFNN)**
- **TextCNN**
- **BiLSTM with Attention**
- **BERT**
- **BioBERT**
- **PubMedBERT**

Each model is tested with text-only and text+features configurations, incorporating sentiment analysis and linguistic feature engineering.

## Project Structure

```
health_misinfo_detection/
├── data/
│   ├── raw/              # Original datasets
│   ├── processed/        # Preprocessed datasets
│   └── embeddings/       # Pre-trained embeddings (GloVe, Word2Vec)
├── src/
│   ├── data/
│   │   ├── loaders.py
│   │   ├── preprocessors.py
│   │   └── datasets.py
│   ├── features/
│   │   ├── sentiment.py
│   │   ├── linguistic.py
│   │   └── embeddings.py
│   ├── models/
│   │   ├── base_model.py
│   │   ├── ffnn.py
│   │   ├── textcnn.py
│   │   ├── bilstm_attention.py
│   │   └── transformers.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── callbacks.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── utils/
│       ├── config.py
│       └── helpers.py
├── experiments/
│   └── configs/          # Experiment configuration files
├── notebooks/            # Jupyter notebooks for exploration
├── results/              # Model outputs and metrics
└── main.py               # Entry point
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Train a BERT model on CoAID dataset
python main.py --model bert --dataset coaid --features

# Cross-dataset evaluation
python main.py --model biobert --train-data coaid --test-data healthfact
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers (HuggingFace)
- NLTK, spaCy
- scikit-learn
- matplotlib, seaborn

## Experiment Design

The system evaluates 48 configurations:
- 6 models × 2 input types × 2 training datasets × 2 test scenarios

## License

MIT License
