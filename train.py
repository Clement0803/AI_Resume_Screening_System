import pandas as pd
import re
import string
import pickle
import os
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ------------------------
# Configuration
# ------------------------
DATA_PATH = "data/Resume.csv"
MODEL_DIR = "model"
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "category_model.pkl")

# ------------------------
# Text Cleaning
# ------------------------
def clean_text(text):
    """
    Clean and normalize text for ML processing
    
    Steps:
    1. Convert to lowercase
    2. Remove email addresses
    3. Remove numbers
    4. Remove punctuation
    5. Remove extra whitespace
    """
    text = text.lower()
    text = re.sub(r'\S+@\S+', ' ', text)  # Remove emails
    text = re.sub(r'\d+', ' ', text)  # Remove numbers
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# ------------------------
# Load and Validate Dataset
# ------------------------
def load_and_validate_data(data_path):
    """Load dataset and perform basic validation"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Validate required columns
    required_columns = ['Resume', 'Category']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for missing values
    missing_resumes = df['Resume'].isna().sum()
    missing_categories = df['Category'].isna().sum()
    
    if missing_resumes > 0 or missing_categories > 0:
        print(f"⚠️  Warning: Found {missing_resumes} missing resumes and {missing_categories} missing categories")
        df = df.dropna(subset=['Resume', 'Category'])
        print(f"✓ Dropped rows with missing values. Remaining: {len(df)} rows")
    
    print(f"✓ Dataset loaded successfully: {len(df)} resumes")
    print(f"✓ Categories: {df['Category'].nunique()} unique categories")
    print("\nCategory distribution:")
    print(df['Category'].value_counts())
    
    return df

# ------------------------
# Main Training Pipeline
# ------------------------
def main():
    print("="*60)
    print("AI RESUME COPILOT - MODEL TRAINING")
    print("="*60)
    
    # Load data
    try:
        df = load_and_validate_data(DATA_PATH)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Clean text
    print("\nCleaning resume text...")
    df['clean_resume'] = df['Resume'].apply(clean_text)
    print("✓ Text cleaning complete")
    
    # ------------------------
    # TF-IDF Vectorization
    # ------------------------
    print("\nTraining TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),  # Use both unigrams and bigrams
        min_df=2,  # Ignore terms that appear in fewer than 2 documents
        max_df=0.8  # Ignore terms that appear in more than 80% of documents
    )
    
    X = tfidf.fit_transform(df['clean_resume'])
    y = df['Category']
    
    print(f"✓ TF-IDF vectorization complete")
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    
    # ------------------------
    # Train-Test Split
    # ------------------------
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        stratify=y, 
        random_state=42
    )
    print(f"✓ Training samples: {X_train.shape[0]}")
    print(f"✓ Testing samples: {X_test.shape[0]}")
    
    # ------------------------
    # Train Classifier
    # ------------------------
    print("\nTraining Logistic Regression classifier...")
    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        multi_class='multinomial',
        solver='lbfgs'
    )
    clf.fit(X_train, y_train)
    print("✓ Classifier training complete")
    
    # ------------------------
    # Evaluate Model
    # ------------------------
    print("\nEvaluating model...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # ------------------------
    # Save Models
    # ------------------------
    print("\nSaving models...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(tfidf, f)
    print(f"✓ TF-IDF vectorizer saved to {TFIDF_PATH}")
    
    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"✓ Classifier saved to {CLASSIFIER_PATH}")
    
    print(f"\n{'='*60}")
    print("✅ MODEL TRAINING COMPLETE!")
    print(f"{'='*60}")
    print("\nYou can now run the application with:")
    print("  streamlit run app.py")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()