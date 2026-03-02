"""
Our Own Sentiment Predictor - Uses locally trained model.
No external APIs - completely self-owned!
"""
import os
import pickle
import logging
import re
from pathlib import Path
from typing import Tuple, Dict, Optional, Set, List

logger = logging.getLogger(__name__)


def _safe_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid float for %s=%r. Using default=%s",
            name,
            raw,
            default
        )
        return default


class OurSentimentPredictor:
    """
    Sentiment predictor using our own trained model.
    This is 100% our own - trained by us, running locally!
    """
    
    def __init__(self, model_dir: str = None):
        """
        Initialize our predictor with our own trained model.
        
        Args:
            model_dir: Directory containing our trained model
        """
        if model_dir is None:
            # Default to models directory in project
            model_dir = Path(__file__).parent / 'models'
        else:
            model_dir = Path(model_dir)
        
        self.model_dir = model_dir
        self.model = None
        self.preprocessor = None
        self.metadata = None
        self.is_loaded = False
        self.known_vocabulary: Set[str] = set()
        self.min_confidence = _safe_env_float('MIN_SENTIMENT_CONFIDENCE', 0.0)
        self.neutral_on_unknown = (
            os.getenv('NEUTRAL_ON_UNKNOWN_VOCAB', 'false').lower() == 'true'
        )
        self.rule_positive_words = {
            'love', 'great', 'excellent', 'awesome', 'happy', 'amazing', 'fantastic',
            'perfect', 'good', 'nice', 'wonderful', 'best', 'satisfied', 'recommend',
            'impressed', 'helpful', 'fast', 'smooth', 'clean', 'enjoy'
        }
        self.rule_negative_words = {
            'bad', 'worst', 'terrible', 'awful', 'hate', 'horrible', 'poor',
            'disappointed', 'broken', 'slow', 'useless', 'scam', 'refund', 'problem',
            'angry', 'crash', 'error', 'fail', 'annoying', 'waste'
        }
        self.emotion_lexicon = {
            'joy': {'happy', 'great', 'awesome', 'love', 'excited', 'amazing', 'wonderful', 'delight'},
            'sadness': {'sad', 'unhappy', 'depressed', 'disappointed', 'upset', 'down'},
            'anger': {'angry', 'hate', 'furious', 'annoyed', 'frustrated', 'irritated', 'rage'},
            'fear': {'afraid', 'scared', 'worried', 'anxious', 'nervous', 'fear'},
            'surprise': {'surprised', 'shocked', 'unexpected', 'wow', 'suddenly'},
            'trust': {'trust', 'reliable', 'secure', 'safe', 'confident', 'dependable'},
        }
        
        # Load our model
        self._load_model()
    
    def _load_model(self):
        """Load our trained model from disk."""
        
        model_path = self.model_dir / 'sentiment_model.pkl'
        preprocessor_path = self.model_dir / 'preprocessor.pkl'
        metadata_path = self.model_dir / 'metadata.json'
        
        try:
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Loaded our model from: {model_path}")
            
            # Load preprocessor
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            logger.info(f"Loaded our preprocessor from: {preprocessor_path}")

            # Cache vocabulary for unknown-text detection
            tfidf = getattr(self.model, 'named_steps', {}).get('tfidf')
            if tfidf is not None and hasattr(tfidf, 'vocabulary_'):
                self.known_vocabulary = set(tfidf.vocabulary_.keys())
            
            # Load metadata
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Model info: {self.metadata}")
            
            self.is_loaded = True
            logger.info("Our sentiment model is ready!")
            
        except FileNotFoundError:
            logger.error("Our model not found! Please run train_model.py first.")
            raise RuntimeError("Model not found. Run train_model.py to create it!")

    def _contains_known_terms(self, processed_text: str) -> bool:
        """Check whether preprocessed text has terms learned during training."""
        if not processed_text:
            return False

        # If vocabulary is unavailable, skip this safety and continue with inference.
        if not self.known_vocabulary:
            return True

        tokens = processed_text.split()
        if not tokens:
            return False

        if any(token in self.known_vocabulary for token in tokens):
            return True

        # Also check bigrams used by TF-IDF ngram_range=(1,2).
        if len(tokens) > 1:
            for i in range(len(tokens) - 1):
                if f"{tokens[i]} {tokens[i + 1]}" in self.known_vocabulary:
                    return True

        return False

    def _apply_decision_policy(
        self,
        prediction: str,
        confidence: float,
        processed_text: str
    ) -> Tuple[str, float, Optional[str]]:
        """
        Convert weak/unknown predictions into neutral to avoid forced labels.
        """
        if not processed_text:
            return "neutral", 0.0, "empty_text"

        if self.neutral_on_unknown and not self._contains_known_terms(processed_text):
            return "neutral", 0.0, "unknown_vocabulary"

        if self.min_confidence > 0 and confidence < self.min_confidence:
            return "neutral", float(confidence), "low_confidence"

        return prediction, float(confidence), None

    def _empty_probabilities(self) -> Dict[str, float]:
        """Build a safe fallback probability distribution."""
        classes = [str(cls) for cls in getattr(self.model, 'classes_', [])]
        if not classes:
            classes = ['negative', 'positive']
        uniform = 1.0 / len(classes)
        return {cls: uniform for cls in classes}

    def _fallback_model_input(self, text: str) -> str:
        """
        Use a lightweight cleaned input when preprocessing removes everything.
        """
        return ' '.join(str(text).lower().split())

    def _tokenize_for_rules(self, text: str) -> List[str]:
        """Tokenize plain text for lexicon-based features."""
        return re.findall(r"[a-zA-Z']+", str(text).lower())

    def rule_based_sentiment(self, text: str) -> Dict[str, float]:
        """Simple lexicon-based baseline model for comparison mode."""
        tokens = self._tokenize_for_rules(text)
        if not tokens:
            return {'sentiment': 'neutral', 'confidence': 0.0, 'score': 0.0}

        pos_hits = sum(1 for token in tokens if token in self.rule_positive_words)
        neg_hits = sum(1 for token in tokens if token in self.rule_negative_words)
        score = pos_hits - neg_hits

        if score > 0:
            sentiment = 'positive'
        elif score < 0:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        confidence = min(1.0, (abs(score) / max(1, len(tokens))) + 0.5 if sentiment != 'neutral' else 0.5)
        return {
            'sentiment': sentiment,
            'confidence': float(round(confidence, 4)),
            'score': float(score)
        }

    def detect_emotions(self, text: str) -> Dict:
        """Rule-based emotion detection with normalized scores."""
        tokens = self._tokenize_for_rules(text)
        scores: Dict[str, int] = {emotion: 0 for emotion in self.emotion_lexicon}
        for token in tokens:
            for emotion, words in self.emotion_lexicon.items():
                if token in words:
                    scores[emotion] += 1

        total_hits = sum(scores.values())
        if total_hits == 0:
            return {
                'top_emotion': 'neutral',
                'scores': {emotion: 0.0 for emotion in scores},
                'matched_terms': []
            }

        normalized = {
            emotion: round(score / total_hits, 4)
            for emotion, score in scores.items()
        }
        top_emotion = max(normalized, key=normalized.get)
        matched_terms = [
            token for token in tokens
            if any(token in words for words in self.emotion_lexicon.values())
        ]
        return {
            'top_emotion': top_emotion,
            'scores': normalized,
            'matched_terms': matched_terms[:20]
        }

    def explain_prediction(self, text: str, top_n: int = 6) -> Dict:
        """
        Provide token-level explainability for model predictions.
        """
        processed_text = self.preprocessor.preprocess(text)
        model_input = processed_text or self._fallback_model_input(text)
        if not model_input:
            return {'top_positive': [], 'top_negative': [], 'model_input': model_input}

        tfidf = self.model.named_steps.get('tfidf')
        classifier = self.model.named_steps.get('classifier')
        if tfidf is None or classifier is None or not hasattr(classifier, 'coef_'):
            return {'top_positive': [], 'top_negative': [], 'model_input': model_input}

        vector = tfidf.transform([model_input])
        feature_names = tfidf.get_feature_names_out()
        coef = classifier.coef_[0]

        row = vector.tocoo()
        contributions = []
        for col_idx, value in zip(row.col, row.data):
            token = str(feature_names[col_idx])
            score = float(value * coef[col_idx])
            contributions.append({'token': token, 'score': round(score, 6)})

        top_positive = sorted(
            [item for item in contributions if item['score'] > 0],
            key=lambda item: item['score'],
            reverse=True
        )[:top_n]
        top_negative = sorted(
            [item for item in contributions if item['score'] < 0],
            key=lambda item: item['score']
        )[:top_n]

        return {
            'model_input': model_input,
            'top_positive': top_positive,
            'top_negative': top_negative
        }
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict sentiment for input text using our model.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (sentiment_label, confidence_score)
        """
        if not self.is_loaded:
            raise RuntimeError("Our model is not loaded!")
        
        if not text or not text.strip():
            return "neutral", 0.0
        
        # Preprocess the text using our preprocessor
        processed_text = self.preprocessor.preprocess(text)
        model_input = processed_text or self._fallback_model_input(text)
        if not model_input:
            return "neutral", 0.0

        if self.neutral_on_unknown and not self._contains_known_terms(model_input):
            return "neutral", 0.0
        
        # Get prediction from our model
        prediction = self.model.predict([model_input])[0]
        
        # Get confidence scores
        proba = self.model.predict_proba([model_input])[0]
        confidence = float(max(proba))
        
        final_prediction, final_confidence, _ = self._apply_decision_policy(
            prediction=str(prediction),
            confidence=confidence,
            processed_text=model_input
        )
        return final_prediction, final_confidence
    
    def predict_detailed(self, text: str) -> Dict:
        """
        Get detailed prediction results.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with detailed prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("Our model is not loaded!")
        
        # Preprocess
        processed_text = self.preprocessor.preprocess(text)
        model_input = processed_text or self._fallback_model_input(text)

        probabilities = self._empty_probabilities()
        raw_prediction = "neutral"
        raw_confidence = 0.0
        decision_reason: Optional[str] = None

        has_known_terms = self._contains_known_terms(model_input)
        if model_input and (has_known_terms or not self.neutral_on_unknown):
            # Get prediction
            raw_prediction = str(self.model.predict([model_input])[0])

            # Get probabilities
            proba = self.model.predict_proba([model_input])[0]
            classes = self.model.classes_

            probabilities = {}
            for i, cls in enumerate(classes):
                probabilities[str(cls)] = float(proba[i])
            raw_confidence = float(max(proba))
            if not has_known_terms:
                decision_reason = "unknown_vocabulary"
        elif self.neutral_on_unknown:
            decision_reason = "unknown_vocabulary"
        elif not processed_text:
            decision_reason = "empty_text"

        prediction, confidence, threshold_reason = self._apply_decision_policy(
            prediction=raw_prediction,
            confidence=raw_confidence,
            processed_text=model_input
        )
        if threshold_reason is not None:
            decision_reason = threshold_reason

        result = {
            'text': text,
            'processed_text': processed_text,
            'prediction': prediction,
            'raw_prediction': raw_prediction,
            'confidence': confidence,
            'raw_confidence': raw_confidence,
            'probabilities': probabilities,
            'decision_reason': decision_reason,
            'min_confidence_threshold': self.min_confidence,
            'emotions': self.detect_emotions(text),
            'explainability': self.explain_prediction(text),
            'baseline_rule_model': self.rule_based_sentiment(text),
            'model_type': 'Our Own TF-IDF + Logistic Regression',
            'model_info': self.metadata
        }
        
        return result
    
    def get_model_info(self) -> Dict:
        """Get information about our model."""
        return {
            'model_name': 'Our Custom Sentiment Model',
            'model_type': 'TF-IDF + Logistic Regression',
            'is_loaded': self.is_loaded,
            'metadata': self.metadata,
            'description': '100% self-owned, locally trained model'
        }
    
    def add_training_data(self, text: str, label: str):
        """
        Add new training data (for future retraining).
        
        Args:
            text: Training text
            label: Sentiment label ('positive' or 'negative')
        """
        # Store for later retraining
        training_data_file = self.model_dir / 'additional_data.txt'
        
        with open(training_data_file, 'a') as f:
            f.write(f"{label}|{text}\n")
        
        logger.info(f"Added new training data: ({label}) {text[:50]}...")


def create_predictor(model_dir: str = None) -> OurSentimentPredictor:
    """
    Factory function to create our predictor.
    
    Args:
        model_dir: Optional model directory
        
    Returns:
        OurSentimentPredictor instance
    """
    return OurSentimentPredictor(model_dir)
