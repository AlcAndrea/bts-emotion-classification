# Dual-Input Emotion Classification in K-Pop Song Lyrics
**Comparing TF-IDF LinearSVC Baselines and Regularized 1D Convolutional Neural Networks with Acoustic Label Grounding**

## 👥 Authors
* **Alcaeus Andrea Conner** 
* **Ashley Manzo**

---

## 📌 Project Overview
This repository contains the complete implementation, experimental pipelines, and evaluation artifacts for our Deep Learning midterm project. We investigate Music Emotion Recognition (MER) using bilingual Korean-English lyrics from the curated BTS track dataset (`bts.csv`). 

To address the common pitfall of noisy or subjective annotations in song lyric datasets, our pipeline implements **Acoustic Label Grounding** via Thayer's bidimensional affective model (mapping Spotify valence and energy features into four distinct affective states: *Happy, Sad, Raring, Calm*). We then compare a shallow sparse baseline (**TF-IDF + LinearSVC**) against a deep multimodal architecture (**Regularized Dual-Input 1D CNN** fusing text embeddings with structured metadata).

---

## 🗂️ Repository Structure
```text
bts-emotion-analysis/
├── data/
│   └── bts.csv                       # Dataset containing tracks, lyrics, and Spotify features
├── src/
│   ├── main.py                       # Main execution script (Preprocessing, Baseline, CNN, Evaluation)
│   └── ...
├── Dual_Input_Emotion_Classification_in_K_Pop_Song_Lyrics.pdf
├── Dual Input Emotion Classification In K-Pop Lyrics PPT
├── learning_curves.png               # Training/validation accuracy & loss curves
├── LinearSVC_Confusion_Matrix.png    # Confusion matrix for TF-IDF + LinearSVC baseline
├── 1D_CNN_Confusion_Matrix.png       # Confusion matrix for Dual-Input 1D CNN
├── requirements.txt                  # Python package dependencies
└── README.md                         # Project documentation