import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Concatenate, Dropout

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
    
    # Engineered Features (Safe extraction checks)
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
    
    # Generate Target Labels (Mock logic for TWC emotion model)
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

def main():
    # 1. Load Data (Ensure your dataset filename matches here)
    bts_df = load_and_preprocess_data("bts.csv")
    print(f"Total tracks processed: {len(bts_df)}")
    
    # 2. Encode & Split
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
    
    # 3. Baseline Model (TF-IDF + LinearSVC)
    print("\n--- Training LinearSVC Baseline ---")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train_tfidf, y_train)
    
    svc = LinearSVC(class_weight='balanced', max_iter=2000, random_state=42)
    svc.fit(X_train_resampled, y_train_resampled)
    y_pred_svc = svc.predict(X_test_tfidf)
    
    print("\nLinearSVC Classification Report:\n")
    print(classification_report(y_test, y_pred_svc, target_names=le.classes_))
    plot_confusion_matrix(y_test, y_pred_svc, le.classes_, 'LinearSVC Confusion Matrix')
    
    # 4. Deep Learning Model (Dual-Input 1D CNN)
    print("\n--- Training Dual-Input 1D CNN ---")
    MAX_VOCAB, MAX_SEQ_LENGTH = 10000, 500
    
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)
    
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=MAX_SEQ_LENGTH, padding='post')
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=MAX_SEQ_LENGTH, padding='post')
    
    # Architecture
    text_input = Input(shape=(MAX_SEQ_LENGTH,), name="text_input")
    x = Embedding(input_dim=MAX_VOCAB, output_dim=128)(text_input)
    x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(64, activation='relu')(x)
    
    meta_input = Input(shape=(len(meta_cols),), name="meta_input")
    y = Dense(16, activation='relu')(meta_input)
    
    z = Concatenate()([x, y])
    z = Dense(64, activation='relu')(z)
    z = Dropout(0.5)(z)
    output = Dense(len(le.classes_), activation='softmax')(z)
    
    cnn_model = Model(inputs=[text_input, meta_input], outputs=output)
    cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    history = cnn_model.fit(
        {'text_input': X_train_seq, 'meta_input': X_train_meta}, y_train,
        validation_data=({'text_input': X_test_seq, 'meta_input': X_test_meta}, y_test),
        epochs=10, batch_size=16, verbose=1
    )
    
    # Plot Learning Curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('CNN Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('CNN Loss')
    plt.legend()
    plt.savefig("learning_curves.png")
    plt.show()

    # CNN Evaluation & Matrix
    cnn_probs = cnn_model.predict({'text_input': X_test_seq, 'meta_input': X_test_meta})
    y_pred_cnn = np.argmax(cnn_probs, axis=1)
    
    print("\nDual-Input 1D CNN Classification Report:\n")
    print(classification_report(y_test, y_pred_cnn, target_names=le.classes_))
    plot_confusion_matrix(y_test, y_pred_cnn, le.classes_, '1D CNN Confusion Matrix', cmap='Purples')

if __name__ == "__main__":
    main()