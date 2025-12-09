# fakenewsdetection
Goal is to design, train, and evaluate machine learning models that detect fake vs. real health information

A Machine Learning Architecture for Detecting Fake Health Information

## Overview

This project implements and compares multiple neural network architectures for detecting health misinformation across two datasets (CoAID and HealthFact). The system evaluates:

- **Feed-Forward Neural Networks (FFNN)**
- **TextCNN**
- **BiLSTM with Attention**
- **BERT**

Each model is tested with text-only and text+features configurations, incorporating sentiment analysis and linguistic feature engineering.

## Datasets 

### 1. CoAID Dataset (COVID-19 Healthcare Misinformation)
- **Source**: https://github.com/cuilimeng/CoAID
- **Description**: COVID-19 related claims, news articles, and social media posts
- **Download Instructions**:
  ```bash
  # Clone the repository or download specific files
  # Place files in: data/raw/coaid/
  ```
- **Files needed**:
  - `NewsRealCOVID-19.csv`
  - `NewsFakeCOVID-19.csv` 
  - Or combined dataset files

### 2. HealthFact Dataset
- **Source**: Multiple options:
  - https://github.com/neemakot/Health-Fact-Checking (HealthVer)
  - https://github.com/clips/healthfc
- **Description**: Health-related claims with fact-checking labels
- **Download Instructions**:
  ```bash
  # Download CSV files
  # Place in: data/raw/healthfact/
  ```

## Manual Download Steps

1. Visit the GitHub repositories above
2. Download the CSV/JSON files
3. Place them in the appropriate subdirectories
4. The loaders will automatically detect and process them

## Alternative: Sample Data

If datasets are unavailable, sample data generators are provided in the loaders.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers (HuggingFace)
- NLTK, spaCy
- scikit-learn
- matplotlib, seaborn

## Experiment Design

The system evaluates 24 configurations:
- 4 models × 2 input types × 2 training datasets × 2 test scenarios
