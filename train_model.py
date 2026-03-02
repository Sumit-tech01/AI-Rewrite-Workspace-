"""
Train our own Sentiment Analysis Model from scratch.
Uses TF-IDF + Logistic Regression - completely self-owned, no external APIs.
"""
import json
import pickle
from pathlib import Path

# ML Libraries - sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Import our preprocessor
from utils import TextPreprocessor

# ==================== TRAINING DATA ====================
# Our own dataset - we can expand this with more samples
TRAINING_DATA = [
    # Positive reviews
    ("I love this product, it's amazing!", "positive"),
    ("This is the best thing I have ever bought", "positive"),
    ("Great quality and fast shipping", "positive"),
    ("Absolutely wonderful experience", "positive"),
    ("Highly recommend to everyone", "positive"),
    ("Fantastic quality, worth the money", "positive"),
    ("Exceeded my expectations", "positive"),
    ("Love it so much!", "positive"),
    ("Perfect in every way", "positive"),
    ("Customer service was excellent", "positive"),
    ("This made my day so much better", "positive"),
    ("Incredible value for money", "positive"),
    ("The quality is outstanding", "positive"),
    ("I'm very happy with my purchase", "positive"),
    ("Five stars all the way", "positive"),
    ("Best purchase I've made", "positive"),
    ("Really impressed with this", "positive"),
    ("Wonderful product, works great", "positive"),
    ("Amazing quality and design", "positive"),
    ("So glad I bought this", "positive"),
    ("This is exactly what I needed", "positive"),
    ("Terrific product!", "positive"),
    ("I am completely satisfied", "positive"),
    ("Can't recommend enough", "positive"),
    ("Simply the best", "positive"),
    
    # Negative reviews
    ("This is the worst product ever", "negative"),
    ("Terrible quality, waste of money", "negative"),
    ("Very disappointed with this purchase", "negative"),
    ("Don't buy this, it's fake", "negative"),
    ("Product broke after one use", "negative"),
    ("Extremely poor quality", "negative"),
    ("Not worth the price at all", "negative"),
    ("Very bad experience", "negative"),
    ("I hate this product", "negative"),
    ("Absolutely terrible", "negative"),
    ("Worst purchase ever made", "negative"),
    ("Save your money, avoid this", "negative"),
    ("Quality is very poor", "negative"),
    ("Not as described at all", "negative"),
    ("Completely useless product", "negative"),
    ("Very unhappy with this", "negative"),
    ("This is a scam", "negative"),
    ("Horrible, would not recommend", "negative"),
    ("Fell apart immediately", "negative"),
    ("Total waste of money", "negative"),
    ("Very disappointing", "negative"),
    ("Product didn't work at all", "negative"),
    ("Avoid this at all costs", "negative"),
    ("Not satisfied at all", "negative"),
    ("Terrible experience", "negative"),
    
    # More positive samples
    ("I really enjoy using this every day", "positive"),
    ("This has made my life so much easier", "positive"),
    ("The design is beautiful", "positive"),
    ("Works perfectly as described", "positive"),
    ("I would buy this again", "positive"),
    ("Excellent value for the price", "positive"),
    ("The best in its category", "positive"),
    ("So easy to use", "positive"),
    ("My friend loved it too", "positive"),
    ("Arrived in perfect condition", "positive"),
    
    # More negative samples
    ("Stopped working after a week", "negative"),
    ("The material feels cheap", "negative"),
    ("Not worth the hype", "negative"),
    ("Wish I could get a refund", "negative"),
    ("The color was different", "negative"),
    ("Smelled weird when I opened it", "negative"),
    ("Too small for what I needed", "negative"),
    ("Customer support was unhelpful", "negative"),
    ("The battery dies too fast", "negative"),
    ("Missing parts when it arrived", "negative"),
]

# ==================== MODEL TRAINING ====================

def train_model():
    """Train our own sentiment model."""
    
    print("=" * 50)
    print("TRAINING OUR OWN SENTIMENT MODEL")
    print("=" * 50)
    
    # Extract texts and labels
    texts = [item[0] for item in TRAINING_DATA]
    labels = [item[1] for item in TRAINING_DATA]
    
    print(f"Total training samples: {len(texts)}")
    print(f"Positive samples: {labels.count('positive')}")
    print(f"Negative samples: {labels.count('negative')}")
    
    # Preprocess texts
    print("\nPreprocessing texts...")
    preprocessor = TextPreprocessor()
    processed_texts = preprocessor.preprocess_batch(texts)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create our pipeline: TF-IDF + Logistic Regression
    print("\nBuilding our model pipeline...")
    
    # TF-IDF Vectorizer - converts text to numerical features
    tfidf = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=1,
        max_df=0.95
    )
    
    # Logistic Regression classifier
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0
    )
    
    # Build pipeline
    model = Pipeline([
        ('tfidf', tfidf),
        ('classifier', classifier)
    ])
    
    # Train the model
    print("\nTraining our model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating our model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.2%}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Test with some examples
    print("\n" + "=" * 50)
    print("TESTING OUR MODEL")
    print("=" * 50)
    
    test_texts = [
        "I love this so much!",
        "Terrible product, waste of time",
        "It's okay, nothing special",
        "Best purchase ever!",
        "Very disappointed",
    ]
    
    for text in test_texts:
        processed = preprocessor.preprocess(text)
        prediction = model.predict([processed])[0]
        proba = model.predict_proba([processed])[0]
        confidence = max(proba) * 100
        print(f"Text: '{text}'")
        print(f"  -> Prediction: {prediction} ({confidence:.1f}% confidence)")
    
    return model, preprocessor


def save_model(model, preprocessor, model_dir):
    """Save our trained model to disk."""
    
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / 'sentiment_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")
    
    # Save preprocessor
    preprocessor_path = model_dir / 'preprocessor.pkl'
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Preprocessor saved to: {preprocessor_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'TF-IDF + Logistic Regression',
        'training_samples': len(TRAINING_DATA),
        'version': '1.0.0',
        'description': 'Our own custom sentiment model'
    }
    metadata_path = model_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    
    return model_path, preprocessor_path


def main():
    """Main training function."""
    
    # Get project directory
    project_dir = Path(__file__).parent
    model_dir = project_dir / 'models'
    
    # Train model
    model, preprocessor = train_model()
    
    # Save model
    save_model(model, preprocessor, model_dir)
    
    print("\n" + "=" * 50)
    print("MODEL TRAINING COMPLETE!")
    print("We've built our own sentiment analysis model!")
    print("=" * 50)


if __name__ == '__main__':
    main()
