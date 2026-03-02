"""
Text preprocessing utilities for our sentiment model.
"""
import logging
import re

try:
    from nltk.corpus import stopwords
except Exception:
    stopwords = None

try:
    from nltk.stem import WordNetLemmatizer
except Exception:
    WordNetLemmatizer = None

logger = logging.getLogger(__name__)

# Fallback stopwords used when NLTK corpora are unavailable.
FALLBACK_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "here", "hers", "him",
    "his", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
    "yourselves",
}


def _load_stop_words():
    """Load NLTK stop words if available; otherwise use fallback list."""
    if stopwords is not None:
        try:
            return set(stopwords.words('english'))
        except LookupError:
            logger.warning(
                "NLTK stopwords corpus not found. Falling back to built-in list."
            )
    return set(FALLBACK_STOP_WORDS)


class TextPreprocessor:
    """Our own text preprocessing pipeline."""
    
    def __init__(self):
        self.stop_words = _load_stop_words()
        self._lemmatization_enabled = WordNetLemmatizer is not None
        self._lemmatizer_warning_logged = False
        self.lemmatizer = WordNetLemmatizer() if self._lemmatization_enabled else None
    
    def _ensure_runtime_defaults(self):
        """
        Ensure attributes exist even for older pickled instances.
        """
        if not hasattr(self, 'stop_words') or self.stop_words is None:
            self.stop_words = _load_stop_words()
        if not hasattr(self, '_lemmatization_enabled'):
            self._lemmatization_enabled = WordNetLemmatizer is not None
        if not hasattr(self, '_lemmatizer_warning_logged'):
            self._lemmatizer_warning_logged = False
        if not hasattr(self, 'lemmatizer'):
            self.lemmatizer = (
                WordNetLemmatizer() if self._lemmatization_enabled else None
            )

    def _safe_lemmatize(self, word):
        """Lemmatize if corpora are available; otherwise return input token."""
        if not self._lemmatization_enabled or self.lemmatizer is None:
            return word

        try:
            return self.lemmatizer.lemmatize(word)
        except Exception as exc:
            self._lemmatization_enabled = False
            if not self._lemmatizer_warning_logged:
                logger.warning(
                    "Lemmatization unavailable (%s). Continuing without lemmatization.",
                    exc.__class__.__name__
                )
                self._lemmatizer_warning_logged = True
            return word
    
    def preprocess(self, text):
        """Clean and preprocess text."""
        self._ensure_runtime_defaults()
        
        if text is None:
            text = ''
        elif not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Lemmatization
        words = text.split()
        words = [
            self._safe_lemmatize(word)
            for word in words
            if word not in self.stop_words and len(word) > 2
        ]
        
        return ' '.join(words)
    
    def preprocess_batch(self, texts):
        """Preprocess a batch of texts."""
        return [self.preprocess(text) for text in texts]
