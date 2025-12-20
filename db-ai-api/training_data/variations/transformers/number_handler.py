"""Number format variations (digits vs words)."""
from typing import List, Dict
import re
from .base import BaseTransformer


class NumberHandler(BaseTransformer):
    """
    Handle number variations in queries.

    Converts between:
    - Digits: "10"
    - Words: "десять"
    - Suffixed: "10-ть" (Ukrainian style)
    """

    # Ukrainian number words
    UK_NUMBERS = {
        '1': 'один', '2': 'два', '3': 'три', '4': 'чотири', '5': "п'ять",
        '6': 'шість', '7': 'сім', '8': 'вісім', '9': "дев'ять", '10': 'десять',
        '11': 'одинадцять', '12': 'дванадцять', '13': 'тринадцять',
        '14': 'чотирнадцять', '15': "п'ятнадцять",
        '20': 'двадцять', '25': 'двадцять п\'ять',
        '30': 'тридцять', '50': "п'ятдесят",
        '100': 'сто', '200': 'двісті', '500': "п'ятсот",
        '1000': 'тисяча',
    }

    # English number words
    EN_NUMBERS = {
        '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
        '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten',
        '11': 'eleven', '12': 'twelve', '13': 'thirteen',
        '14': 'fourteen', '15': 'fifteen',
        '20': 'twenty', '25': 'twenty five',
        '30': 'thirty', '50': 'fifty',
        '100': 'hundred', '200': 'two hundred', '500': 'five hundred',
        '1000': 'thousand',
    }

    # Reverse mappings (word to digit)
    UK_WORDS_TO_NUM = {v: k for k, v in UK_NUMBERS.items()}
    EN_WORDS_TO_NUM = {v: k for k, v in EN_NUMBERS.items()}

    # Common TOP N values in business queries
    COMMON_TOP_N = [5, 10, 15, 20, 50, 100]

    @property
    def name(self) -> str:
        return "number_handler"

    def transform(self, text: str, language: str = "uk") -> List[str]:
        """
        Generate number format variations.

        Args:
            text: Input text
            language: Language code

        Returns:
            List of number variations
        """
        results = []

        # Expand digits to words
        results.extend(self.digits_to_words(text, language))

        # Expand words to digits
        results.extend(self.words_to_digits(text, language))

        # Generate Ukrainian-style suffixed numbers
        if language == "uk":
            results.extend(self._generate_suffixed(text))

        # Generate TOP N variations
        results.extend(self._generate_top_n_variations(text, language))

        # Remove duplicates and original
        results = list(set(results) - {text, text.lower()})

        return results

    def digits_to_words(self, text: str, language: str = "uk") -> List[str]:
        """
        Convert digits to word form.

        "топ 10 товарів" → "топ десять товарів"
        """
        number_map = self.UK_NUMBERS if language == "uk" else self.EN_NUMBERS
        results = []

        # Find all numbers in text
        for digit, word in number_map.items():
            pattern = r'\b' + digit + r'\b'
            if re.search(pattern, text):
                new_text = re.sub(pattern, word, text)
                if new_text != text:
                    results.append(new_text)

        return results

    def words_to_digits(self, text: str, language: str = "uk") -> List[str]:
        """
        Convert word numbers to digits.

        "топ десять товарів" → "топ 10 товарів"
        """
        word_map = self.UK_WORDS_TO_NUM if language == "uk" else self.EN_WORDS_TO_NUM
        results = []
        text_lower = text.lower()

        for word, digit in word_map.items():
            if word in text_lower:
                new_text = text_lower.replace(word, digit)
                if new_text != text_lower:
                    results.append(new_text)

        return results

    def _generate_suffixed(self, text: str) -> List[str]:
        """
        Generate Ukrainian-style suffixed numbers.

        "10" → "10-ть", "5" → "5-ть"
        """
        results = []

        # Match standalone numbers
        pattern = r'\b(\d+)\b'

        def add_suffix(match):
            num = match.group(1)
            # Common Ukrainian suffix patterns
            return f"{num}-ть"

        new_text = re.sub(pattern, add_suffix, text)
        if new_text != text:
            results.append(new_text)

        return results

    def _generate_top_n_variations(
        self,
        text: str,
        language: str = "uk"
    ) -> List[str]:
        """
        Generate variations with different TOP N values.

        "топ 10 товарів" → ["топ 5 товарів", "топ 20 товарів", ...]
        """
        results = []

        # Patterns for TOP N queries
        if language == "uk":
            patterns = [
                r'(топ|перші|найкращі)\s*(\d+)',
                r'(\d+)\s*(найкращих|перших|топ)',
            ]
        else:
            patterns = [
                r'(top|first|best)\s*(\d+)',
                r'(\d+)\s*(best|first|top)',
            ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Replace with different N values
                for n in self.COMMON_TOP_N:
                    # Determine which group has the number
                    if match.group(2) and match.group(2).isdigit():
                        # Pattern 1: keyword + number
                        new_text = re.sub(
                            pattern,
                            lambda m: f"{m.group(1)} {n}",
                            text,
                            flags=re.IGNORECASE
                        )
                    else:
                        # Pattern 2: number + keyword
                        new_text = re.sub(
                            pattern,
                            lambda m: f"{n} {m.group(2)}",
                            text,
                            flags=re.IGNORECASE
                        )

                    if new_text != text:
                        results.append(new_text)

        return results[:5]  # Limit results

    def can_transform(self, text: str, language: str = "uk") -> bool:
        """Check if text contains numbers to transform."""
        # Contains digits
        if any(c.isdigit() for c in text):
            return True

        # Contains number words
        word_map = self.UK_WORDS_TO_NUM if language == "uk" else self.EN_WORDS_TO_NUM
        text_lower = text.lower()
        return any(word in text_lower for word in word_map)
