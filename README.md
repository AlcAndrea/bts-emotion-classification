# Emotion Classification in Bilingual K-Pop Lyrics

A Master's-level machine learning and natural language processing pipeline designed to classify the emotional sentiment of bilingual (Korean-English) song lyrics. This project uses a hybrid architecture combining a TF-IDF LinearSVC baseline with a custom **Dual-Input 1D Convolutional Neural Network (CNN)** that jointly processes textual n-grams and engineered song metadata.

---

## 🚀 Project Overview
Analyzing K-Pop lyrics presents a unique NLP challenge due to dense **code-switching**—the rapid alternation between Korean (Hangul) and English within the same track. This mixing splits the semantic feature space, requiring models that can capture both linguistic nuance and contextual metadata (such as release year and streaming popularity metrics).

### Key Features:
- **Bilingual Text Cleaning:** Strips structural Genius tags (`[Verse]`, `[Chorus]`, etc.) while preserving Korean and English characters.
- **Language Filtering:** Excludes Japanese tracks to maintain a strict dual-language feature space.
- **Data Balancing:** Employs **Random Oversampling (ROS)** to mitigate class imbalance across emotional quadrants (*Raring, Calm, Sad, Happy*).
- **Two-Model Comparison:** Contrasts a lightweight baseline (**TF-IDF + LinearSVC**) against an advanced **Dual-Input 1D CNN** (featuring parallel text sequence and metadata dense branches).
- **Rigorous Evaluation:** Generates stratified train-test splits (80/20) with strict data leakage prevention, complete with learning curves and confusion matrices.

---

## 🛠️ Tech Stack & Dependencies
- **Language:** Python 3.13
- **Deep Learning:** TensorFlow / Keras (Functional API)
- **Machine Learning:** Scikit-learn, Imbalanced-learn (RandomOverSampler)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

---

## 📁 Repository Structure
```text
├── bts.csv                   # Raw dataset containing tracks, lyrics, and metadata
├── main.py                   # Main executable pipeline (preprocessing, modeling, evaluation)
├── requirements.txt          # Pinned package dependencies for reproducibility
├── .gitignore                # Git exclusion rules for virtual environments and artifacts
├── learning_curves.png       # Training vs. validation accuracy/loss plots
└── 1D_CNN_Confusion_Matrix.png # Themed purple confusion matrix for model evaluation

Here is a clean, professional `README.md` file tailored for your GitHub repository. It highlights your Master's level engineering approach, the dual-input architecture, and the bilingual K-Pop lyric analysis methodology.

Create a new file in your VS Code workspace named `README.md`, paste this text into it, and commit it along with your other files!

```markdown
# Emotion Classification in Bilingual K-Pop Lyrics

A Master's-level machine learning and natural language processing pipeline designed to classify the emotional sentiment of bilingual (Korean-English) song lyrics. This project uses a hybrid architecture combining a TF-IDF LinearSVC baseline with a custom **Dual-Input 1D Convolutional Neural Network (CNN)** that jointly processes textual n-grams and engineered song metadata.

---

## 🚀 Project Overview
Analyzing K-Pop lyrics presents a unique NLP challenge due to dense **code-switching**—the rapid alternation between Korean (Hangul) and English within the same track. This mixing splits the semantic feature space, requiring models that can capture both linguistic nuance and contextual metadata (such as release year and streaming popularity metrics).

### Key Features:
- **Bilingual Text Cleaning:** Strips structural Genius tags (`[Verse]`, `[Chorus]`, etc.) while preserving Korean and English characters.
- **Language Filtering:** Excludes Japanese tracks to maintain a strict dual-language feature space.
- **Data Balancing:** Employs **Random Oversampling (ROS)** to mitigate class imbalance across emotional quadrants (*Raring, Calm, Sad, Happy*).
- **Two-Model Comparison:** Contrasts a lightweight baseline (**TF-IDF + LinearSVC**) against an advanced **Dual-Input 1D CNN** (featuring parallel text sequence and metadata dense branches).
- **Rigorous Evaluation:** Generates stratified train-test splits (80/20) with strict data leakage prevention, complete with learning curves and confusion matrices.

---

## 🛠️ Tech Stack & Dependencies
- **Language:** Python 3.13
- **Deep Learning:** TensorFlow / Keras (Functional API)
- **Machine Learning:** Scikit-learn, Imbalanced-learn (RandomOverSampler)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

---

## 📁 Repository Structure
```text
├── bts.csv                   # Raw dataset containing tracks, lyrics, and metadata
├── main.py                   # Main executable pipeline (preprocessing, modeling, evaluation)
├── requirements.txt          # Pinned package dependencies for reproducibility
├── .gitignore                # Git exclusion rules for virtual environments and artifacts
├── learning_curves.png       # Training vs. validation accuracy/loss plots
└── 1D_CNN_Confusion_Matrix.png # Themed purple confusion matrix for model evaluation

```

---

## ⚙️ Setup & Installation

1. **Clone the Repository:**
```bash
git clone [https://github.com/your-username/bts-emotion-classification.git](https://github.com/your-username/bts-emotion-classification.git)
cd bts-emotion-classification

```


2. **Create and Activate a Virtual Environment:**
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

```


3. **Install Dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt

```


4. **Run the Pipeline:**
Ensure your dataset (`bts.csv`) is in the root directory, then execute:
```bash
python main.py

```



---

## 📊 Model Architecture

The advanced deep learning model utilizes a **Dual-Input Functional API** structure:

1. **Text Branch:** Tokenized sequences pass through an Embedding layer, a 1D Convolutional layer (`filters=128`, `kernel_size=5`), Global Max Pooling, and a Dense feature extraction layer.
2. **Metadata Branch:** Engineered numerical features (`year`, `views`, `is_korean`, `is_english`) are normalized using `StandardScaler` and processed through a dense sub-network.
3. **Feature Fusion:** Text and metadata representations are concatenated, passed through dense regularization with Dropout (`0.5`), and mapped to emotion classes via Softmax activation.

---

## 💡 Results & Insights

* **LinearSVC Baseline:** Offers fast, robust convergence on sparse TF-IDF spaces, resisting overfitting on smaller datasets.
* **Dual-Input 1D CNN:** Successfully captures localized n-gram patterns across code-switched phrases, though heavily influenced by dominant English choruses during max-pooling phases.

```

```
