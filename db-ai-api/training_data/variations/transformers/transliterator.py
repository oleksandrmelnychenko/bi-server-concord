"""Ukrainian transliteration (Cyrillic <-> Latin).

Handles common transliteration patterns users type, like "prodazhi" instead of "продажі".
"""
from typing import List, Dict, Optional
from itertools import product
from functools import lru_cache
from .base import BaseTransformer


# Module-level caching for transliteration operations
@lru_cache(maxsize=1024)
def _cached_looks_transliterated(text: str) -> bool:
    """Cached check if text looks like transliterated Ukrainian."""
    text_lower = text.lower()

    if not all(c.isascii() or c.isspace() for c in text_lower):
        return False

    uk_patterns = [
        'zh', 'kh', 'shch', 'sch', 'ch', 'sh', 'ts',
        'ya', 'ia', 'yu', 'iu', 'yi', 'ye', 'ie',
    ]
    pattern_count = sum(1 for p in uk_patterns if p in text_lower)
    if pattern_count >= 2:
        return True

    uk_word_patterns = [
        'prodazh', 'tovar', 'klient', 'borh', 'zakas',
        'skilk', 'pokaz', 'vivest', 'porahuy',
    ]
    return any(pattern in text_lower for pattern in uk_word_patterns)


@lru_cache(maxsize=1024)
def _cached_transliterate_to_ukrainian(text: str) -> Optional[str]:
    """Cached Latin to Ukrainian transliteration."""
    if not _cached_looks_transliterated(text):
        return None

    text_lower = text.lower()
    result = []
    i = 0

    while i < len(text_lower):
        matched = False
        for length in [4, 3, 2]:
            if i + length <= len(text_lower):
                chunk = text_lower[i:i + length]
                if chunk in _LATIN_TO_UK:
                    result.append(_LATIN_TO_UK[chunk])
                    i += length
                    matched = True
                    break

        if not matched:
            char = text_lower[i]
            result.append(_LATIN_TO_UK.get(char, char))
            i += 1

    return ''.join(result)


# Module-level Latin to Ukrainian mapping for cached function
_LATIN_TO_UK = {
    'shch': 'щ', 'sch': 'щ',
    'zh': 'ж', 'kh': 'х', 'ts': 'ц', 'ch': 'ч', 'sh': 'ш',
    'ya': 'я', 'ia': 'я', 'ja': 'я',
    'yu': 'ю', 'iu': 'ю', 'ju': 'ю',
    'ye': 'є', 'ie': 'є', 'je': 'є',
    'yi': 'ї', 'ji': 'ї',
    'ph': 'ф',
    'a': 'а', 'b': 'б', 'c': 'ц', 'd': 'д', 'e': 'е',
    'f': 'ф', 'g': 'г', 'h': 'х', 'i': 'і', 'j': 'й',
    'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н', 'o': 'о',
    'p': 'п', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у',
    'v': 'в', 'w': 'в', 'x': 'кс', 'y': 'и', 'z': 'з', 'q': 'к',
}


class Transliterator(BaseTransformer):
    """
    Convert Ukrainian text to Latin transliteration and vice versa.

    Supports multiple transliteration styles:
    - Official Ukrainian (passport standard)
    - Informal/common (what users actually type)
    """

    # Official Ukrainian transliteration (Ukrainian passport standard)
    UK_TO_LATIN_OFFICIAL = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'h', 'ґ': 'g',
        'д': 'd', 'е': 'e', 'є': 'ie', 'ж': 'zh', 'з': 'z',
        'и': 'y', 'і': 'i', 'ї': 'i', 'й': 'i', 'к': 'k',
        'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p',
        'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f',
        'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
        'ь': '', 'ю': 'iu', 'я': 'ia', "'": '',
    }

    # Informal transliteration variants (what users commonly type)
    # Some letters have multiple common representations
    UK_TO_LATIN_INFORMAL = {
        'а': ['a'],
        'б': ['b'],
        'в': ['v', 'w'],
        'г': ['h', 'g'],
        'ґ': ['g'],
        'д': ['d'],
        'е': ['e'],
        'є': ['ye', 'ie', 'je', 'e'],
        'ж': ['zh', 'j'],
        'з': ['z'],
        'и': ['y', 'i', 'u'],  # Very variable
        'і': ['i', 'y'],
        'ї': ['yi', 'i', 'ji', 'ii'],
        'й': ['y', 'j', 'i'],
        'к': ['k'],
        'л': ['l'],
        'м': ['m'],
        'н': ['n'],
        'о': ['o'],
        'п': ['p'],
        'р': ['r'],
        'с': ['s'],
        'т': ['t'],
        'у': ['u'],
        'ф': ['f', 'ph'],
        'х': ['kh', 'h', 'x'],
        'ц': ['ts', 'c', 'tz'],
        'ч': ['ch'],
        'ш': ['sh'],
        'щ': ['shch', 'sch', 'sh'],
        'ь': ['', "'"],
        'ю': ['yu', 'iu', 'ju', 'u'],
        'я': ['ya', 'ia', 'ja', 'a'],
        "'": [''],
    }

    # Latin to Ukrainian (reverse, for detecting transliterated input)
    LATIN_TO_UK_PATTERNS = {
        # Multi-character patterns (check first)
        'shch': 'щ', 'sch': 'щ',
        'zh': 'ж', 'kh': 'х', 'ts': 'ц', 'ch': 'ч', 'sh': 'ш',
        'ya': 'я', 'ia': 'я', 'ja': 'я',
        'yu': 'ю', 'iu': 'ю', 'ju': 'ю',
        'ye': 'є', 'ie': 'є', 'je': 'є',
        'yi': 'ї', 'ji': 'ї',
        'ph': 'ф',
        # Single character patterns
        'a': 'а', 'b': 'б', 'c': 'ц', 'd': 'д', 'e': 'е',
        'f': 'ф', 'g': 'г', 'h': 'х', 'i': 'і', 'j': 'й',
        'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н', 'o': 'о',
        'p': 'п', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у',
        'v': 'в', 'w': 'в', 'x': 'кс', 'y': 'и', 'z': 'з', 'q': 'к',
    }

    @property
    def name(self) -> str:
        return "transliterator"

    def transform(self, text: str, language: str = "uk") -> List[str]:
        """
        Generate transliteration variations.

        For Ukrainian: convert to Latin transliterations
        For Latin: convert to Ukrainian (if it looks transliterated)

        Args:
            text: Input text
            language: Source language

        Returns:
            List of transliterated variations
        """
        results = []

        if language == "uk":
            # Ukrainian to Latin transliterations
            results.extend(self.transliterate_to_latin(text))
        else:
            # Try to convert Latin to Ukrainian (if it looks transliterated)
            ukrainian = self.transliterate_to_ukrainian(text)
            if ukrainian and ukrainian != text:
                results.append(ukrainian)

        return results

    def transliterate_to_latin(
        self,
        text: str,
        style: str = "all",
        max_variants: int = 5
    ) -> List[str]:
        """
        Convert Ukrainian text to Latin transliteration.

        Args:
            text: Ukrainian text
            style: 'official', 'informal', or 'all'
            max_variants: Maximum number of variants to generate

        Returns:
            List of transliterated variants
        """
        results = set()

        # Official transliteration
        if style in ('official', 'all'):
            official = self._apply_mapping(text.lower(), self.UK_TO_LATIN_OFFICIAL)
            results.add(official)

        # Informal variants
        if style in ('informal', 'all'):
            informal_variants = self._generate_informal_variants(
                text.lower(),
                max_variants=max_variants
            )
            results.update(informal_variants)

        return list(results)

    def transliterate_to_ukrainian(self, text: str) -> Optional[str]:
        """
        Convert Latin transliteration back to Ukrainian (cached).

        Args:
            text: Potentially transliterated Latin text

        Returns:
            Ukrainian text or None if not transliterated
        """
        return _cached_transliterate_to_ukrainian(text)

    def detect_transliteration(self, text: str) -> bool:
        """Check if text appears to be Ukrainian transliterated to Latin (cached)."""
        return _cached_looks_transliterated(text)

    def _apply_mapping(self, text: str, mapping: Dict[str, str]) -> str:
        """Apply simple character mapping."""
        result = []
        for char in text:
            result.append(mapping.get(char, char))
        return ''.join(result)

    def _generate_informal_variants(
        self,
        text: str,
        max_variants: int = 5
    ) -> List[str]:
        """
        Generate informal transliteration variants.

        Uses the variant mappings to create multiple possible transliterations.
        """
        # Get options for each character
        char_options = []
        for char in text:
            if char in self.UK_TO_LATIN_INFORMAL:
                char_options.append(self.UK_TO_LATIN_INFORMAL[char])
            else:
                char_options.append([char])

        # Limit combinations to avoid explosion
        # Calculate total combinations
        total = 1
        for opts in char_options:
            total *= len(opts)
            if total > 1000:
                break

        if total > max_variants * 10:
            # Too many combinations, use sampling strategy
            return self._sample_variants(char_options, max_variants)

        # Generate all combinations up to max
        results = set()
        for combo in product(*char_options):
            results.add(''.join(combo))
            if len(results) >= max_variants:
                break

        return list(results)

    def _sample_variants(
        self,
        char_options: List[List[str]],
        count: int
    ) -> List[str]:
        """Sample variants when there are too many combinations."""
        import random
        results = set()

        # Always include first option (most common)
        results.add(''.join(opts[0] for opts in char_options))

        # Sample random combinations
        for _ in range(count * 10):
            variant = ''.join(random.choice(opts) for opts in char_options)
            results.add(variant)
            if len(results) >= count:
                break

        return list(results)

    def _looks_transliterated(self, text: str) -> bool:
        """Check if text looks like Ukrainian transliterated to Latin (cached)."""
        return _cached_looks_transliterated(text)

    def can_transform(self, text: str, language: str = "uk") -> bool:
        """Check if text can be meaningfully transformed."""
        if language == "uk":
            # Has Cyrillic characters
            return any('\u0400' <= c <= '\u04ff' for c in text)
        else:
            # Looks like transliterated Ukrainian (cached)
            return _cached_looks_transliterated(text)


# Common Ukrainian words and their transliterations
COMMON_TRANSLITERATIONS = {
    # Products
    'товари': ['tovary', 'tovari', 'tovarі'],
    'товар': ['tovar'],
    'продукти': ['produkty', 'produkti'],
    'продукція': ['produktsia', 'produkciya', 'produkcija'],
    # Sales
    'продажі': ['prodazhi', 'prodaji', 'prodazhy'],
    'продаж': ['prodazh', 'prodaj'],
    # Customers
    'клієнти': ['klienty', 'kliyenty', 'klijenty'],
    'клієнт': ['klient', 'kliyent'],
    'покупці': ['pokuptsi', 'pokupci'],
    # Financial
    'борги': ['borhy', 'borgi', 'borgi'],
    'борг': ['borh', 'borg'],
    'заборгованість': ['zaborhovanist', 'zaborgovanist'],
    # Actions
    'показати': ['pokazaty', 'pokazati'],
    'покажи': ['pokazhy', 'pokaji', 'pokazy'],
    'скільки': ['skilky', 'skilki'],
    'порахуй': ['porahuy', 'porahuj', 'porakhuy'],
    # Common
    'топ': ['top'],
    'запаси': ['zapasy', 'zapasi'],
    'замовлення': ['zamovlennia', 'zamovlennya', 'zamovlenya'],
}
