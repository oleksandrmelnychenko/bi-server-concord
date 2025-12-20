"""Mixed Ukrainian-English query generation.

Common in Ukrainian IT/business context where users mix languages:
- "Show топ 10 товарів"
- "Покажи best customers"
- "Top 10 клієнтів"
"""
from typing import List, Dict, Tuple
import re
from .base import BaseTransformer


class LanguageMixer(BaseTransformer):
    """
    Generate mixed Ukrainian-English query variations.

    Handles common code-switching patterns in Ukrainian business context.
    """

    # Common word pairs (Ukrainian -> English and vice versa)
    WORD_PAIRS = {
        # Actions
        'покажи': 'show',
        'показати': 'show',
        'виведи': 'list',
        'знайди': 'find',
        'порахуй': 'count',
        'дай': 'get',
        # Nouns
        'товари': 'products',
        'товарів': 'products',
        'клієнти': 'customers',
        'клієнтів': 'customers',
        'продажі': 'sales',
        'продажів': 'sales',
        'борги': 'debts',
        'боргів': 'debts',
        'замовлення': 'orders',
        'замовлень': 'orders',
        'залишки': 'stock',
        'платежі': 'payments',
        # Adjectives/Quantities
        'топ': 'top',
        'найкращі': 'best',
        'перші': 'first',
        'всі': 'all',
        'активні': 'active',
    }

    # Reverse mapping
    EN_TO_UK = {v: k for k, v in WORD_PAIRS.items()}

    # Common mixed patterns
    MIXING_PATTERNS = [
        # English command + Ukrainian noun
        ('en_command', 'uk_noun'),
        # Ukrainian command + English noun
        ('uk_command', 'en_noun'),
        # Keep numbers, mix rest
        ('mixed_with_numbers',),
    ]

    @property
    def name(self) -> str:
        return "language_mixer"

    def transform(self, text: str, language: str = "uk") -> List[str]:
        """
        Generate mixed language variations.

        Args:
            text: Input text
            language: Source language

        Returns:
            List of mixed language variations
        """
        return self.generate_mixed(text, language)

    def generate_mixed(
        self,
        text: str,
        source_language: str = "uk",
        max_variations: int = 5
    ) -> List[str]:
        """
        Generate mixed language variations.

        Args:
            text: Input query
            source_language: Primary language of input
            max_variations: Maximum variations to generate

        Returns:
            List of mixed variations
        """
        results = set()
        text_lower = text.lower()
        words = text_lower.split()

        if source_language == "uk":
            # Mix Ukrainian with English
            results.update(self._mix_uk_with_en(words, text_lower))
        else:
            # Mix English with Ukrainian
            results.update(self._mix_en_with_uk(words, text_lower))

        # Remove original
        results.discard(text_lower)
        results.discard(text)

        return list(results)[:max_variations]

    def _mix_uk_with_en(self, words: List[str], original: str) -> List[str]:
        """Generate variations by replacing Ukrainian words with English."""
        results = []

        # Strategy 1: Replace first word (usually command)
        if words and words[0] in self.WORD_PAIRS:
            new_words = words.copy()
            new_words[0] = self.WORD_PAIRS[words[0]]
            results.append(' '.join(new_words))

        # Strategy 2: Replace nouns (last meaningful word)
        for i in range(len(words) - 1, -1, -1):
            word = words[i]
            if word in self.WORD_PAIRS and not word.isdigit():
                new_words = words.copy()
                new_words[i] = self.WORD_PAIRS[word]
                results.append(' '.join(new_words))
                break

        # Strategy 3: Replace adjectives like "топ" → "top"
        for i, word in enumerate(words):
            if word in ['топ', 'найкращі', 'перші']:
                new_words = words.copy()
                new_words[i] = self.WORD_PAIRS.get(word, word)
                results.append(' '.join(new_words))

        # Strategy 4: Full hybrid - EN command + UK noun
        if len(words) >= 2:
            first_en = self.WORD_PAIRS.get(words[0], words[0])
            # Keep rest as Ukrainian
            hybrid = [first_en] + words[1:]
            results.append(' '.join(hybrid))

        return results

    def _mix_en_with_uk(self, words: List[str], original: str) -> List[str]:
        """Generate variations by replacing English words with Ukrainian."""
        results = []

        # Similar strategies but reversed
        for i, word in enumerate(words):
            if word in self.EN_TO_UK:
                new_words = words.copy()
                new_words[i] = self.EN_TO_UK[word]
                results.append(' '.join(new_words))

        return results

    def detect_language_mix(self, text: str) -> Dict[str, float]:
        """
        Detect the language composition of text.

        Returns:
            Dict with 'uk' and 'en' percentages
        """
        text_lower = text.lower()
        words = [w for w in text_lower.split() if w.isalpha()]

        if not words:
            return {'uk': 0.0, 'en': 0.0}

        uk_count = 0
        en_count = 0

        for word in words:
            # Check if Cyrillic
            if any('\u0400' <= c <= '\u04ff' for c in word):
                uk_count += 1
            # Check if Latin
            elif any('a' <= c <= 'z' for c in word):
                en_count += 1

        total = len(words)
        return {
            'uk': uk_count / total,
            'en': en_count / total,
            'mixed': 1.0 if (uk_count > 0 and en_count > 0) else 0.0
        }

    def is_mixed_language(self, text: str) -> bool:
        """Check if text contains both Ukrainian and English."""
        composition = self.detect_language_mix(text)
        return composition['mixed'] > 0

    def can_transform(self, text: str, language: str = "uk") -> bool:
        """Check if text can be transformed."""
        # Has words that can be mixed
        text_lower = text.lower()
        pairs = self.WORD_PAIRS if language == "uk" else self.EN_TO_UK
        return any(word in text_lower for word in pairs)
