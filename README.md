# BTS Lyric Emotion Classification

This repository contains our Deep Neural Networks midterm project exploring sentiment and emotion classification on bilingual (Korean-English) BTS song lyrics. The project compares a traditional TF-IDF + LinearSVC baseline against a regularized Dual-Input 1D Convolutional Neural Network (CNN) that fuses text embeddings with track metadata.

## Overview & Approach

Analyzing K-Pop lyrics is tricky due to code-switching (the mixing of Korean and English within the same track), which often fragments standard text representations. To tackle this, my pipeline does the following:
- **Preprocessing:** Cleans Genius tags (`[Verse]`, `[Chorus]`, etc.), filters out Japanese tracks, and extracts language indicators.
- **Baseline:** Uses a sparse TF-IDF feature extractor paired with a balanced `LinearSVC`.
- **Deep Model:** Implements a Keras Functional 1D CNN that processes tokenized lyric sequences alongside scaled metadata (`year`, `views`, language flags).
- **Overfitting Fixes:** Added L2 regularization, spatial dropout, and an early stopping callback (`val_loss` patience) to keep training and validation metrics stable on our ~389-track dataset.


## Project Structure

- `main.py`: Full execution script (data loading, preprocessing, model training, evaluation, and plotting).
- `bts.csv`: Dataset containing tracks, lyrics, and metadata.
- `requirements.txt`: Python package dependencies.
- `learning_curves.png`: Training vs. validation loss/accuracy plots.
- `1D_CNN_Confusion_Matrix.png` & `LinearSVC_Confusion_Matrix.png`: Evaluation confusion matrices.

## How to Run

1. Clone the repository and navigate into the folder.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt