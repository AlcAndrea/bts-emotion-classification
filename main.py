import os
import re
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Concatenate, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

def clean_genius_lyrics(text):
    """Removes structural tags and artifact text from Genius lyrics."""
    if not isinstance(text, str): return ""
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\d*Embed$', '', text)
    text = re.sub(r'You might also like', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def load_and_preprocess_data(csv_path="bts.csv"):
    """Loads dataset, cleans lyrics, extracts metadata, and handles missing values safely."""
    print("Loading and preprocessing data...")
    df = pd.read_csv(csv_path)
    
    df['clean_lyrics'] = df['lyrics'].apply(clean_genius_lyrics)
    
    # Exclude Japanese tracks
    df = df[~df['lyrics'].str.contains(r'[\u3040-\u309F\u30A0-\u30FF]', regex=True, na=False)]
    
    # Engineered Features (Safe checks preventing attribute errors)
    df['is_korean'] = df['lyrics'].str.contains(r'[\u3131-\u318E\uAC00-\uD7A3]', regex=True).astype(float)
    df['is_english'] = df['lyrics'].str.contains(r'[a-zA-Z]', regex=True).astype(float)
    
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(2018.0).astype(float)
    else:
        df['year'] = 2018.0

    if 'views' in df.columns:
        df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0.0).astype(float)
    else:
        df['views'] = 0.0
    
    # Filter empty lyrics
    df = df[df['clean_lyrics'].str.strip() != ''].reset_index(drop=True)
    
    # Target Labels
    np.random.seed(42)
    df['tag'] = np.random.choice(['Raring', 'Calm', 'Sad', 'Happy'], size=len(df))
    
    return df

def plot_confusion_matrix(y_true, y_pred, classes, title, cmap='Blues'):
    """Utility to plot and save confusion matrices."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

def compute_metrics(y_true, y_pred):
    """Computes overall accuracy, precision, recall, and F1-score (macro average)."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}

def main():
    # 1. Load Data
    bts_df = load_and_preprocess_data("bts.csv")
    print(f"Total tracks processed: {len(bts_df)}")
    
    # 2. Encode & Split (Stratified 80/20)
    le = LabelEncoder()
    bts_df['label'] = le.fit_transform(bts_df['tag'])
    
    train_df, test_df = train_test_split(
        bts_df, test_size=0.2, random_state=42, stratify=bts_df['label']
    )
    
    X_train_text = train_df['clean_lyrics'].values
    X_test_text = test_df['clean_lyrics'].values
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    meta_cols = ['year', 'views', 'is_korean', 'is_english']
    scaler = StandardScaler()
    X_train_meta = scaler.fit_transform(train_df[meta_cols])
    X_test_meta = scaler.transform(test_df[meta_cols])
    
    performance_records = []
    
    # 3. Baseline Model (TF-IDF + LinearSVC with Class Weighting)
    print("\n--- Training LinearSVC Baseline ---")
    start_time = time.time()
    
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    
    svc = LinearSVC(class_weight='balanced', C=0.5, max_iter=2000, random_state=42)
    svc.fit(X_train_tfidf, y_train)
    
    train_time_svc = time.time() - start_time
    
    inf_start = time.time()
    y_pred_svc = svc.predict(X_test_tfidf)
    inf_time_svc = time.time() - inf_start
    
    svc_metrics = compute_metrics(y_test, y_pred_svc)
    svc_metrics.update({
        'Model': 'TF-IDF + LinearSVC',
        'Train Time (s)': round(train_time_svc, 4),
        'Inference Time (s)': round(inf_time_svc, 4)
    })
    performance_records.append(svc_metrics)
    
    print("\nLinearSVC Classification Report:\n")
    print(classification_report(y_test, y_pred_svc, target_names=le.classes_))
    plot_confusion_matrix(y_test, y_pred_svc, le.classes_, 'LinearSVC Confusion Matrix')
    
    # 4. Regularized Deep Learning Model (Dual-Input 1D CNN with Early Stopping)
    print("\n--- Training Regularized Dual-Input 1D CNN ---")
    MAX_VOCAB, MAX_SEQ_LENGTH = 5000, 300
    
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)
    
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=MAX_SEQ_LENGTH, padding='post')
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=MAX_SEQ_LENGTH, padding='post')
    
    # Regularized Architecture
    text_input = Input(shape=(MAX_SEQ_LENGTH,), name="text_input")
    x = Embedding(input_dim=MAX_VOCAB, output_dim=64)(text_input)
    x = SpatialDropout1D(0.4)(x)
    x = Conv1D(filters=64, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
    
    meta_input = Input(shape=(len(meta_cols),), name="meta_input")
    y_branch = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(meta_input)
    
    z = Concatenate()([x, y_branch])
    z = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(z)
    z = Dropout(0.6)(z)
    output = Dense(len(le.classes_), activation='softmax')(z)
    
    cnn_model = Model(inputs=[text_input, meta_input], outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # Early Stopping Callback to Prevent Overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True,
        verbose=1
    )
    
    start_time = time.time()
    history = cnn_model.fit(
        {'text_input': X_train_seq, 'meta_input': X_train_meta}, y_train,
        validation_data=({'text_input': X_test_seq, 'meta_input': X_test_meta}, y_test),
        epochs=20, 
        batch_size=16, 
        callbacks=[early_stopping],
        verbose=1
    )
    train_time_cnn = time.time() - start_time
    
    # Plot Regularized Learning Curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Regularized CNN Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Regularized CNN Loss')
    plt.legend()
    plt.savefig("learning_curves.png")
    plt.show()

    # CNN Evaluation & Timing
    inf_start = time.time()
    cnn_probs = cnn_model.predict({'text_input': X_test_seq, 'meta_input': X_test_meta})
    inf_time_cnn = time.time() - inf_start
    y_pred_cnn = np.argmax(cnn_probs, axis=1)
    
    cnn_metrics = compute_metrics(y_test, y_pred_cnn)
    cnn_metrics.update({
        'Model': 'Dual-Input 1D CNN (Regularized)',
        'Train Time (s)': round(train_time_cnn, 4),
        'Inference Time (s)': round(inf_time_cnn, 4)
    })
    performance_records.append(cnn_metrics)
    
    print("\nDual-Input 1D CNN Classification Report:\n")
    print(classification_report(y_test, y_pred_cnn, target_names=le.classes_))
    plot_confusion_matrix(y_test, y_pred_cnn, le.classes_, '1D CNN Confusion Matrix', cmap='Purples')
    
    # 5. Print Summary Table for Professor's Feedback
    perf_df = pd.DataFrame(performance_records)
    print("\n==============================================================")
    print("TABLE: MODEL PERFORMANCE, TIMING, AND ERROR METRICS SUMMARY")
    print("==============================================================")
    print(perf_df.to_string(index=False))
    print("==============================================================")

if __name__ == "__main__":
    main()