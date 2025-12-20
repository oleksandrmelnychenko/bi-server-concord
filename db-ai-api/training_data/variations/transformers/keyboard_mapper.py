"""Keyboard layout error simulation and correction.

This is CRITICAL for Ukrainian users who often forget to switch keyboard layout.
When typing Ukrainian text with English layout active, "товари" becomes "njdfhb".
"""
from typing import List, Optional, Tuple
from functools import lru_cache
from .base import BaseTransformer


# Module-level caching for keyboard layout operations
@lru_cache(maxsize=1024)
def _cached_apply_layout(text: str, layout_type: str) -> str:
    """Cached layout transformation."""
    mapping = _UK_TO_EN if layout_type == "uk_to_en" else _EN_TO_UK
    return ''.join(mapping.get(char, char) for char in text)


@lru_cache(maxsize=1024)
def _cached_detect_uk_in_en(text: str) -> bool:
    """Cached detection of Ukrainian typed with English layout."""
    # Must be mostly ASCII
    ascii_count = sum(1 for c in text if c.isascii())
    if ascii_count < len(text) * 0.8:
        return False

    # Check common patterns
    text_lower = text.lower()
    uk_patterns_in_en = [
        'njdfhb', 'ghjlf;', 'rkstyn', 'pfrfp', 'crfkmrb',
        'gjrfpfnb', 'njg', 'ljhub',
    ]
    for pattern in uk_patterns_in_en:
        if pattern in text_lower:
            return True

    # Check if conversion produces Cyrillic
    converted = _cached_apply_layout(text, "en_to_uk")
    cyrillic_count = sum(1 for c in converted if '\u0400' <= c <= '\u04ff')
    return cyrillic_count > len(converted) * 0.5


@lru_cache(maxsize=1024)
def _cached_detect_en_in_uk(text: str) -> bool:
    """Cached detection of English typed with Ukrainian layout."""
    cyrillic_count = sum(1 for c in text if '\u0400' <= c <= '\u04ff')
    if cyrillic_count < len(text) * 0.3:
        return False

    text_lower = text.lower()
    en_patterns_in_uk = ['ыргц', 'ещз', 'дшые', 'сщгте']
    for pattern in en_patterns_in_uk:
        if pattern in text_lower:
            return True

    return False


@lru_cache(maxsize=1024)
def _cached_correct_layout(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Cached layout correction. Returns (corrected_text, error_type)."""
    if _cached_detect_uk_in_en(text):
        return (_cached_apply_layout(text, "en_to_uk"), "uk_in_en")
    if _cached_detect_en_in_uk(text):
        return (_cached_apply_layout(text, "uk_to_en"), "en_in_uk")
    return (None, None)


# Module-level layout mappings for cached functions
_UK_TO_EN = {
    # Top row
    'й': 'q', 'ц': 'w', 'у': 'e', 'к': 'r', 'е': 't',
    'н': 'y', 'г': 'u', 'ш': 'i', 'щ': 'o', 'з': 'p',
    'х': '[', 'ї': ']',
    # Middle row
    'ф': 'a', 'і': 's', 'в': 'd', 'а': 'f', 'п': 'g',
    'р': 'h', 'о': 'j', 'л': 'k', 'д': 'l', 'ж': ';',
    'є': "'",
    # Bottom row
    'я': 'z', 'ч': 'x', 'с': 'c', 'м': 'v', 'и': 'b',
    'т': 'n', 'ь': 'm', 'б': ',', 'ю': '.',
    # Special
    'ґ': '`', "'": '\\',
    # Uppercase
    'Й': 'Q', 'Ц': 'W', 'У': 'E', 'К': 'R', 'Е': 'T',
    'Н': 'Y', 'Г': 'U', 'Ш': 'I', 'Щ': 'O', 'З': 'P',
    'Х': '{', 'Ї': '}',
    'Ф': 'A', 'І': 'S', 'В': 'D', 'А': 'F', 'П': 'G',
    'Р': 'H', 'О': 'J', 'Л': 'K', 'Д': 'L', 'Ж': ':',
    'Є': '"',
    'Я': 'Z', 'Ч': 'X', 'С': 'C', 'М': 'V', 'И': 'B',
    'Т': 'N', 'Ь': 'M', 'Б': '<', 'Ю': '>',
}
_EN_TO_UK = {v: k for k, v in _UK_TO_EN.items()}


class KeyboardMapper(BaseTransformer):
    """
    Simulate and correct keyboard layout errors.

    Common scenario: User types Ukrainian word but keyboard is set to English layout.
    Example: "товари" typed as "njdfhb"
    """

    # Ukrainian ЙЦУКЕН layout -> English QWERTY layout (same physical key positions)
    UK_TO_EN_LAYOUT = {
        # Top row
        'й': 'q', 'ц': 'w', 'у': 'e', 'к': 'r', 'е': 't',
        'н': 'y', 'г': 'u', 'ш': 'i', 'щ': 'o', 'з': 'p',
        'х': '[', 'ї': ']',
        # Middle row
        'ф': 'a', 'і': 's', 'в': 'd', 'а': 'f', 'п': 'g',
        'р': 'h', 'о': 'j', 'л': 'k', 'д': 'l', 'ж': ';',
        'є': "'",
        # Bottom row
        'я': 'z', 'ч': 'x', 'с': 'c', 'м': 'v', 'и': 'b',
        'т': 'n', 'ь': 'm', 'б': ',', 'ю': '.',
        # Special
        'ґ': '`', "'": '\\',
        # Uppercase
        'Й': 'Q', 'Ц': 'W', 'У': 'E', 'К': 'R', 'Е': 'T',
        'Н': 'Y', 'Г': 'U', 'Ш': 'I', 'Щ': 'O', 'З': 'P',
        'Х': '{', 'Ї': '}',
        'Ф': 'A', 'І': 'S', 'В': 'D', 'А': 'F', 'П': 'G',
        'Р': 'H', 'О': 'J', 'Л': 'K', 'Д': 'L', 'Ж': ':',
        'Є': '"',
        'Я': 'Z', 'Ч': 'X', 'С': 'C', 'М': 'V', 'И': 'B',
        'Т': 'N', 'Ь': 'M', 'Б': '<', 'Ю': '>',
    }

    # Reverse mapping: English -> Ukrainian
    EN_TO_UK_LAYOUT = {v: k for k, v in UK_TO_EN_LAYOUT.items()}

    @property
    def name(self) -> str:
        return "keyboard_mapper"

    def transform(self, text: str, language: str = "uk") -> List[str]:
        """
        Generate keyboard layout error variations.

        For Ukrainian text: generate what it would look like typed with English layout.
        For English text: generate what it would look like typed with Ukrainian layout.

        Args:
            text: Input text
            language: Source language ('uk' or 'en')

        Returns:
            List with original + keyboard-error version
        """
        results = []

        if language == "uk":
            # Ukrainian text typed with English keyboard
            error_version = self._apply_layout(text, self.UK_TO_EN_LAYOUT)
            if error_version != text:
                results.append(error_version)
        else:
            # English text typed with Ukrainian keyboard
            error_version = self._apply_layout(text, self.EN_TO_UK_LAYOUT)
            if error_version != text:
                results.append(error_version)

        return results

    def correct_layout_error(self, text: str) -> Optional[str]:
        """
        Attempt to correct a keyboard layout error (cached).

        If text looks like Ukrainian typed with English layout, convert to Ukrainian.
        If text looks like English typed with Ukrainian layout, convert to English.

        Args:
            text: Potentially mistyped text

        Returns:
            Corrected text or None if no correction needed
        """
        corrected, _ = _cached_correct_layout(text)
        return corrected

    def detect_layout_error(self, text: str) -> Optional[str]:
        """
        Detect if text has a keyboard layout error (cached).

        Returns:
            'uk_in_en' if Ukrainian typed with English layout
            'en_in_uk' if English typed with Ukrainian layout
            None if no error detected
        """
        _, error_type = _cached_correct_layout(text)
        return error_type

    def _apply_layout(self, text: str, mapping: dict) -> str:
        """Apply character mapping to text."""
        result = []
        for char in text:
            result.append(mapping.get(char, char))
        return ''.join(result)

    def _looks_like_uk_in_en_layout(self, text: str) -> bool:
        """
        Check if text looks like Ukrainian typed with English keyboard.

        Heuristics:
        - Contains only ASCII letters and common punctuation
        - When converted to Ukrainian, produces valid-looking Ukrainian words
        - Contains character sequences common in Ukrainian but rare in English
        """
        # Must be mostly ASCII
        ascii_count = sum(1 for c in text if c.isascii())
        if ascii_count < len(text) * 0.8:
            return False

        # Common patterns when Ukrainian is typed with English layout
        # These are sequences that would be common Ukrainian words
        uk_patterns_in_en = [
            'njdfhb',   # товари
            'ghjlf;',   # продаж
            'rkstyn',   # клієнт
            'pfrfp',    # заказ
            'crfkmrb',  # скільки
            'gjrfpfnb', # показати
            'njg',      # топ
            'ljhub',    # борги
        ]

        text_lower = text.lower()
        for pattern in uk_patterns_in_en:
            if pattern in text_lower:
                return True

        # Check if conversion produces Cyrillic
        converted = self._apply_layout(text, self.EN_TO_UK_LAYOUT)
        cyrillic_count = sum(1 for c in converted if '\u0400' <= c <= '\u04ff')
        return cyrillic_count > len(converted) * 0.5

    def _looks_like_en_in_uk_layout(self, text: str) -> bool:
        """
        Check if text looks like English typed with Ukrainian keyboard.

        Heuristics:
        - Contains Cyrillic characters
        - When converted to English, produces valid-looking English words
        """
        # Must have some Cyrillic
        cyrillic_count = sum(1 for c in text if '\u0400' <= c <= '\u04ff')
        if cyrillic_count < len(text) * 0.3:
            return False

        # Common patterns when English is typed with Ukrainian layout
        # "show" -> "ыргц", "top" -> "ещз", etc.
        en_patterns_in_uk = [
            'ыргц',   # show
            'ещз',    # top
            'дшые',   # list
            'сщгте',  # count
        ]

        text_lower = text.lower()
        for pattern in en_patterns_in_uk:
            if pattern in text_lower:
                return True

        return False

    def can_transform(self, text: str, language: str = "uk") -> bool:
        """Check if text can be meaningfully transformed."""
        if language == "uk":
            # Has Cyrillic characters
            return any('\u0400' <= c <= '\u04ff' for c in text)
        else:
            # Has ASCII letters
            return any(c.isascii() and c.isalpha() for c in text)


# Common Ukrainian words and their English-layout equivalents (for quick lookup)
COMMON_UK_WORDS_EN_LAYOUT = {
    # Products
    'njdfhb': 'товари',
    'njdfh': 'товар',
    'ghjlernb': 'продукти',
    'ghjlerw': 'продукт',
    'ghjlerws': 'продукці',
    # Sales
    'ghjlf;s': 'продажі',
    'ghjlf;': 'продаж',
    'htfkspfws': 'реалізаці',
    # Customers
    'rkstynb': 'клієнти',
    'rkstyn': 'клієнт',
    'gjregws': 'покупці',
    'pfvjdybrb': 'замовники',
    # Financial
    'ljhub': 'борги',
    'ljhu': 'борг',
    'pf,jhujdfysc': 'заборгованіс',
    'lt,snjhrf': 'дебіторка',
    # Actions
    'gjrfpfnb': 'показати',
    'gjrf;b': 'покажи',
    'dsclb': 'виведи',
    'crjkmrb': 'скільки',
    'gjhf[eq': 'порахуй',
    # Common
    'njg': 'топ',
    'pfgfcs': 'запаси',
    'pfrfp': 'заказ',
    'pfvjdktyyz': 'замовлення',
}
