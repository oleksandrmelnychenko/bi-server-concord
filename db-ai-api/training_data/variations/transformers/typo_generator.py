"""Typo generation for realistic user input simulation."""
from typing import List, Dict
import random
from .base import BaseTransformer


class TypoGenerator(BaseTransformer):
    """
    Generate realistic typos based on keyboard proximity and common errors.

    Typo types:
    - Missing character: "товари" → "товри", "тоари"
    - Adjacent key swap: pressing nearby key
    - Double character: "товари" → "товвари"
    - Transposition: "товари" → "товраи" (swapped adjacent chars)
    """

    # Adjacent keys on Ukrainian ЙЦУКЕН keyboard
    UK_ADJACENT_KEYS = {
        # Row 1
        'й': ['ц', '1', '2'],
        'ц': ['й', 'у', '2', '3'],
        'у': ['ц', 'к', '3', '4'],
        'к': ['у', 'е', '4', '5'],
        'е': ['к', 'н', '5', '6'],
        'н': ['е', 'г', '6', '7'],
        'г': ['н', 'ш', '7', '8'],
        'ш': ['г', 'щ', '8', '9'],
        'щ': ['ш', 'з', '9', '0'],
        'з': ['щ', 'х', '0', '-'],
        'х': ['з', 'ї', '-', '='],
        'ї': ['х', '='],
        # Row 2
        'ф': ['й', 'і', 'ц'],
        'і': ['ф', 'в', 'й', 'ц', 'у'],
        'в': ['і', 'а', 'ц', 'у', 'к'],
        'а': ['в', 'п', 'у', 'к', 'е'],
        'п': ['а', 'р', 'к', 'е', 'н'],
        'р': ['п', 'о', 'е', 'н', 'г'],
        'о': ['р', 'л', 'н', 'г', 'ш'],
        'л': ['о', 'д', 'г', 'ш', 'щ'],
        'д': ['л', 'ж', 'ш', 'щ', 'з'],
        'ж': ['д', 'є', 'щ', 'з', 'х'],
        'є': ['ж', 'з', 'х', 'ї'],
        # Row 3
        'я': ['ф', 'ч', 'і'],
        'ч': ['я', 'с', 'ф', 'і', 'в'],
        'с': ['ч', 'м', 'і', 'в', 'а'],
        'м': ['с', 'и', 'в', 'а', 'п'],
        'и': ['м', 'т', 'а', 'п', 'р'],
        'т': ['и', 'ь', 'п', 'р', 'о'],
        'ь': ['т', 'б', 'р', 'о', 'л'],
        'б': ['ь', 'ю', 'о', 'л', 'д'],
        'ю': ['б', 'л', 'д', 'ж'],
    }

    # Adjacent keys on English QWERTY keyboard
    EN_ADJACENT_KEYS = {
        'q': ['w', 'a', 's', '1', '2'],
        'w': ['q', 'e', 'a', 's', 'd', '2', '3'],
        'e': ['w', 'r', 's', 'd', 'f', '3', '4'],
        'r': ['e', 't', 'd', 'f', 'g', '4', '5'],
        't': ['r', 'y', 'f', 'g', 'h', '5', '6'],
        'y': ['t', 'u', 'g', 'h', 'j', '6', '7'],
        'u': ['y', 'i', 'h', 'j', 'k', '7', '8'],
        'i': ['u', 'o', 'j', 'k', 'l', '8', '9'],
        'o': ['i', 'p', 'k', 'l', '9', '0'],
        'p': ['o', 'l', '0', '-'],
        'a': ['q', 'w', 's', 'z', 'x'],
        's': ['q', 'w', 'e', 'a', 'd', 'z', 'x', 'c'],
        'd': ['w', 'e', 'r', 's', 'f', 'x', 'c', 'v'],
        'f': ['e', 'r', 't', 'd', 'g', 'c', 'v', 'b'],
        'g': ['r', 't', 'y', 'f', 'h', 'v', 'b', 'n'],
        'h': ['t', 'y', 'u', 'g', 'j', 'b', 'n', 'm'],
        'j': ['y', 'u', 'i', 'h', 'k', 'n', 'm'],
        'k': ['u', 'i', 'o', 'j', 'l', 'm'],
        'l': ['i', 'o', 'p', 'k'],
        'z': ['a', 's', 'x'],
        'x': ['a', 's', 'd', 'z', 'c'],
        'c': ['s', 'd', 'f', 'x', 'v'],
        'v': ['d', 'f', 'g', 'c', 'b'],
        'b': ['f', 'g', 'h', 'v', 'n'],
        'n': ['g', 'h', 'j', 'b', 'm'],
        'm': ['h', 'j', 'k', 'n'],
    }

    @property
    def name(self) -> str:
        return "typo_generator"

    def transform(self, text: str, language: str = "uk") -> List[str]:
        """
        Generate typo variations.

        Args:
            text: Input text
            language: Language code

        Returns:
            List of typo variations
        """
        return self.generate_typos(text, language)

    def generate_typos(
        self,
        text: str,
        language: str = "uk",
        typo_types: List[str] = None,
        max_typos: int = 5
    ) -> List[str]:
        """
        Generate typo variations.

        Args:
            text: Input text
            language: Language code
            typo_types: Types of typos to generate
            max_typos: Maximum number of typos to generate

        Returns:
            List of typo variations
        """
        typo_types = typo_types or [
            'missing_char',
            'adjacent_swap',
            'double_char',
            'transposition'
        ]

        results = set()
        adjacent_map = (
            self.UK_ADJACENT_KEYS if language == "uk"
            else self.EN_ADJACENT_KEYS
        )

        if 'missing_char' in typo_types:
            results.update(self._generate_missing_char(text)[:2])

        if 'adjacent_swap' in typo_types:
            results.update(self._generate_adjacent_swap(text, adjacent_map)[:2])

        if 'double_char' in typo_types:
            results.update(self._generate_double_char(text)[:2])

        if 'transposition' in typo_types:
            results.update(self._generate_transposition(text)[:2])

        # Remove original if present
        results.discard(text)
        results.discard(text.lower())

        return list(results)[:max_typos]

    def _generate_missing_char(self, text: str) -> List[str]:
        """
        Remove each character one at a time.

        "товари" → ["овари", "твари", "тоари", "товри", "товаи", "товар"]
        """
        results = []
        for i, char in enumerate(text):
            if char.isalpha():
                typo = text[:i] + text[i + 1:]
                if len(typo) >= 3:  # Ensure result is meaningful
                    results.append(typo)
        return results

    def _generate_adjacent_swap(
        self,
        text: str,
        adjacent_map: Dict[str, List[str]]
    ) -> List[str]:
        """
        Replace characters with adjacent keys.

        "товари" → ["еовари", "тлвари", ...]
        """
        results = []
        for i, char in enumerate(text):
            char_lower = char.lower()
            if char_lower in adjacent_map:
                adjacent = adjacent_map[char_lower]
                if adjacent:
                    # Pick first adjacent key
                    replacement = adjacent[0]
                    if char.isupper():
                        replacement = replacement.upper()
                    typo = text[:i] + replacement + text[i + 1:]
                    results.append(typo)
        return results

    def _generate_double_char(self, text: str) -> List[str]:
        """
        Double random characters.

        "товари" → ["ттовари", "тоовари", "товвари", ...]
        """
        results = []
        for i, char in enumerate(text):
            if char.isalpha():
                typo = text[:i] + char + text[i:]
                results.append(typo)
        return results

    def _generate_transposition(self, text: str) -> List[str]:
        """
        Swap adjacent characters.

        "товари" → ["отвари", "твоари", "тоавр", "товраи", "товаір"]
        """
        results = []
        for i in range(len(text) - 1):
            if text[i].isalpha() and text[i + 1].isalpha():
                typo = text[:i] + text[i + 1] + text[i] + text[i + 2:]
                results.append(typo)
        return results

    def can_transform(self, text: str, language: str = "uk") -> bool:
        """Check if text can have typos generated."""
        # Need at least 3 characters for meaningful typos
        return len(text) >= 3 and any(c.isalpha() for c in text)
