"""Formality level transformation for Ukrainian queries.

Ukrainian has distinct verb forms for different formality levels:
- Casual: покажи, дай, скинь (imperative informal)
- Standard: показати, вивести (infinitive)
- Formal: відобразити, надати (formal infinitive)
"""
from typing import List, Dict
from .base import BaseTransformer


class FormalityShifter(BaseTransformer):
    """
    Transform queries between formality levels.

    Critical for Ukrainian where verb forms differ significantly between
    casual conversation and formal requests.
    """

    # Formality mappings: word -> {level: [alternatives]}
    FORMALITY_MAPPINGS = {
        # Show/Display
        'покажи': {
            'casual': ['покажи', 'дай', 'скинь', 'кинь', 'глянь'],
            'standard': ['показати', 'вивести', 'отримати'],
            'formal': ['відобразити', 'надати', 'представити', 'продемонструвати'],
        },
        'показати': {
            'casual': ['покажи', 'дай', 'скинь'],
            'standard': ['показати', 'вивести'],
            'formal': ['відобразити', 'надати', 'представити'],
        },

        # Count/Calculate
        'скільки': {
            'casual': ['скільки', 'скіки', 'скока'],
            'standard': ['яка кількість', 'який обсяг', 'кількість'],
            'formal': ['яким є показник', 'яке значення має', 'визначити кількість'],
        },
        'порахуй': {
            'casual': ['порахуй', 'полічи', 'глянь скільки'],
            'standard': ['підрахувати', 'порахувати', 'обчислити'],
            'formal': ['визначити', 'обчислити значення', 'встановити кількість'],
        },

        # Find/Search
        'знайди': {
            'casual': ['знайди', 'шукай', 'відшукай', 'глянь'],
            'standard': ['знайти', 'пошук', 'відшукати'],
            'formal': ['здійснити пошук', 'виконати пошук', 'знайти інформацію'],
        },

        # List/Output
        'виведи': {
            'casual': ['виведи', 'дай список', 'покажи'],
            'standard': ['вивести', 'отримати список', 'перелічити'],
            'formal': ['сформувати перелік', 'надати список', 'відобразити перелік'],
        },

        # Give/Provide
        'дай': {
            'casual': ['дай', 'скинь', 'кинь'],
            'standard': ['надати', 'отримати'],
            'formal': ['надати інформацію', 'забезпечити дані'],
        },

        # Question words
        'які': {
            'casual': ['які', 'шо за'],
            'standard': ['які', 'котрі'],
            'formal': ['які саме', 'котрі з'],
        },
        'хто': {
            'casual': ['хто', 'хто там'],
            'standard': ['хто', 'котрий'],
            'formal': ['який клієнт', 'котрий з'],
        },
    }

    # Phrases that indicate formality level
    CASUAL_MARKERS = ['скинь', 'кинь', 'глянь', 'дай', 'скіки', 'шо']
    FORMAL_MARKERS = ['відобразити', 'надати', 'здійснити', 'визначити', 'сформувати']

    @property
    def name(self) -> str:
        return "formality_shifter"

    def transform(self, text: str, language: str = "uk") -> List[str]:
        """
        Generate formality level variations.

        Args:
            text: Input text
            language: Language code (mainly supports 'uk')

        Returns:
            List of formality variations
        """
        if language != "uk":
            return []

        return self.shift_formality(text, target_level="all")

    def shift_formality(
        self,
        text: str,
        target_level: str = "all",
        max_variations: int = 5
    ) -> List[str]:
        """
        Generate formality-shifted variations.

        Args:
            text: Input query
            target_level: 'casual', 'standard', 'formal', or 'all'
            max_variations: Maximum variations to generate

        Returns:
            List of formality-shifted variations
        """
        results = set()
        text_lower = text.lower()

        # Determine current formality level
        current_level = self.detect_formality(text)

        # Determine target levels
        if target_level == "all":
            target_levels = ['casual', 'standard', 'formal']
            # Remove current level to avoid duplicates
            if current_level in target_levels:
                target_levels.remove(current_level)
        else:
            target_levels = [target_level]

        # Apply transformations
        for base_word, levels in self.FORMALITY_MAPPINGS.items():
            if base_word in text_lower:
                for level in target_levels:
                    if level in levels:
                        for replacement in levels[level][:2]:  # Limit per level
                            new_text = text_lower.replace(base_word, replacement)
                            if new_text != text_lower:
                                results.add(new_text)
                                if len(results) >= max_variations:
                                    return list(results)

        return list(results)

    def detect_formality(self, text: str) -> str:
        """
        Detect the formality level of the text.

        Returns:
            'casual', 'standard', or 'formal'
        """
        text_lower = text.lower()

        # Check for formal markers
        for marker in self.FORMAL_MARKERS:
            if marker in text_lower:
                return 'formal'

        # Check for casual markers
        for marker in self.CASUAL_MARKERS:
            if marker in text_lower:
                return 'casual'

        # Default to standard
        return 'standard'

    def to_casual(self, text: str) -> List[str]:
        """Convert to casual formality level."""
        return self.shift_formality(text, target_level='casual')

    def to_formal(self, text: str) -> List[str]:
        """Convert to formal formality level."""
        return self.shift_formality(text, target_level='formal')

    def can_transform(self, text: str, language: str = "uk") -> bool:
        """Check if text can be transformed."""
        if language != "uk":
            return False

        text_lower = text.lower()
        return any(word in text_lower for word in self.FORMALITY_MAPPINGS)
