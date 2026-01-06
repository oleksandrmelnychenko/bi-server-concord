"""Question form transformation for queries.

Transforms between different ways of asking the same question:
- "скільки клієнтів?" → "яка кількість клієнтів?" → "порахуй клієнтів"
- "покажи товари" → "список товарів" → "виведи товари"
"""
from typing import List, Dict, Optional
import re
from .base import BaseTransformer


class QuestionFormer(BaseTransformer):
    """
    Transform between different question forms.

    Handles variations in how users phrase the same intent:
    - Interrogative: "скільки?"
    - Imperative: "порахуй"
    - Noun phrase: "кількість"
    """

    # Question patterns by intent type
    QUESTION_PATTERNS = {
        # Count queries
        'count': {
            'uk': [
                'скільки {noun}',
                'яка кількість {noun}',
                'порахуй {noun}',
                'кількість {noun}',
                '{noun} скільки',
                'загальна кількість {noun}',
            ],
            'en': [
                'how many {noun}',
                'count {noun}',
                '{noun} count',
                'total {noun}',
                'number of {noun}',
                'what is the count of {noun}',
            ]
        },
        # Show/list queries
        'show': {
            'uk': [
                'покажи {noun}',
                'виведи {noun}',
                'список {noun}',
                '{noun}',
                'дай {noun}',
                'відобрази {noun}',
            ],
            'en': [
                'show {noun}',
                'list {noun}',
                'display {noun}',
                'get {noun}',
                '{noun}',
                'retrieve {noun}',
            ]
        },
        # Top N queries
        'top': {
            'uk': [
                'топ {n} {noun}',
                'найкращі {n} {noun}',
                '{n} найкращих {noun}',
                'перші {n} {noun}',
                '{n} перших {noun}',
                'лідери {noun}',
            ],
            'en': [
                'top {n} {noun}',
                'best {n} {noun}',
                '{n} best {noun}',
                'first {n} {noun}',
                '{n} leading {noun}',
            ]
        },
        # Search queries
        'find': {
            'uk': [
                'знайди {noun}',
                'пошук {noun}',
                'де {noun}',
                'відшукай {noun}',
                '{noun} де',
            ],
            'en': [
                'find {noun}',
                'search {noun}',
                'where is {noun}',
                'locate {noun}',
                'look for {noun}',
            ]
        },
        # Comparison queries
        'compare': {
            'uk': [
                'порівняй {noun}',
                '{noun} порівняння',
                'різниця {noun}',
                '{noun} vs',
            ],
            'en': [
                'compare {noun}',
                '{noun} comparison',
                'difference {noun}',
                '{noun} vs',
            ]
        },
    }

    # Patterns to detect query intent
    INTENT_PATTERNS = {
        'count': {
            'uk': [r'скільки', r'кількість', r'порахуй', r'порахувати', r'полічи'],
            'en': [r'how many', r'count', r'total', r'number of'],
        },
        'show': {
            'uk': [r'покажи', r'показати', r'виведи', r'вивести', r'список', r'дай'],
            'en': [r'show', r'list', r'display', r'get', r'retrieve'],
        },
        'top': {
            'uk': [r'топ\s*\d+', r'найкращі', r'перші\s*\d+', r'лідери'],
            'en': [r'top\s*\d+', r'best', r'first\s*\d+', r'leading'],
        },
        'find': {
            'uk': [r'знайди', r'знайти', r'пошук', r'де\s', r'відшукай'],
            'en': [r'find', r'search', r'where', r'locate', r'look for'],
        },
    }

    @property
    def name(self) -> str:
        return "question_former"

    def transform(self, text: str, language: str = "uk") -> List[str]:
        """
        Generate question form variations.

        Args:
            text: Input query
            language: Language code

        Returns:
            List of question form variations
        """
        # Detect intent
        intent = self.detect_intent(text, language)
        if not intent:
            return []

        # Extract noun phrase
        noun = self._extract_noun_phrase(text, language)
        if not noun:
            return []

        # Extract number if present (for top N queries)
        n = self._extract_number(text)

        # Generate variations
        return self._generate_variations(intent, noun, n, language, text)

    def detect_intent(self, text: str, language: str = "uk") -> Optional[str]:
        """
        Detect the query intent.

        Returns:
            Intent type: 'count', 'show', 'top', 'find', or None
        """
        text_lower = text.lower()
        patterns = self.INTENT_PATTERNS

        for intent, lang_patterns in patterns.items():
            for pattern in lang_patterns.get(language, []):
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return intent

        return None

    def _extract_noun_phrase(self, text: str, language: str) -> Optional[str]:
        """Extract the main noun phrase from the query."""
        text_lower = text.lower()

        # Remove known action words to find noun
        action_words = {
            'uk': [
                'покажи', 'показати', 'виведи', 'вивести', 'скільки',
                'порахуй', 'знайди', 'дай', 'топ', 'найкращі', 'перші',
                'яка', 'кількість', 'список', 'відобрази',
            ],
            'en': [
                'show', 'list', 'display', 'get', 'how', 'many', 'count',
                'find', 'search', 'top', 'best', 'first', 'the', 'total',
            ]
        }

        words = text_lower.split()
        # Remove action words and numbers
        noun_words = [
            w for w in words
            if w not in action_words.get(language, [])
            and not w.isdigit()
        ]

        return ' '.join(noun_words) if noun_words else None

    def _extract_number(self, text: str) -> Optional[int]:
        """Extract number from query (for top N)."""
        match = re.search(r'\d+', text)
        return int(match.group()) if match else None

    def _generate_variations(
        self,
        intent: str,
        noun: str,
        n: Optional[int],
        language: str,
        original: str
    ) -> List[str]:
        """Generate variations based on intent and extracted parts."""
        patterns = self.QUESTION_PATTERNS.get(intent, {}).get(language, [])
        if not patterns:
            return []

        results = set()
        for pattern in patterns:
            if '{n}' in pattern:
                if n:
                    variation = pattern.format(noun=noun, n=n)
                else:
                    continue
            else:
                variation = pattern.format(noun=noun)

            # Clean up and add
            variation = ' '.join(variation.split())  # Remove extra spaces
            if variation != original.lower():
                results.add(variation)

        return list(results)[:5]  # Limit variations

    def can_transform(self, text: str, language: str = "uk") -> bool:
        """Check if text can be transformed."""
        return self.detect_intent(text, language) is not None
