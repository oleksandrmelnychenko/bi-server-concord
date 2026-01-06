"""Main variation generator orchestrating all transformers."""
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass, field
from enum import Enum
import random
from pathlib import Path


class TransformationType(Enum):
    """Types of transformations available."""
    KEYBOARD_LAYOUT = "keyboard_layout"
    TRANSLITERATION = "transliteration"
    SYNONYM = "synonym"
    FORMALITY = "formality"
    TYPO = "typo"
    NUMBER = "number"
    QUESTION_FORM = "question_form"
    LANGUAGE_MIX = "language_mix"


@dataclass
class VariationConfig:
    """Configuration for variation generation."""
    # Transformation toggles
    include_keyboard_errors: bool = True
    include_transliteration: bool = True
    include_synonyms: bool = True
    include_formality: bool = True
    include_typos: bool = True
    include_numbers: bool = True
    include_question_forms: bool = True
    include_language_mix: bool = True

    # Limits
    max_variations_per_template: int = 100
    representative_sample_size: int = 15

    # Probabilities for random sampling
    typo_probability: float = 0.3
    keyboard_error_probability: float = 0.2


@dataclass
class QueryVariation:
    """A single query variation."""
    original_id: str
    variation_id: str
    text: str
    language: str  # 'uk', 'en', 'mixed'
    transformation_types: List[TransformationType] = field(default_factory=list)
    is_representative: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class VariationGenerator:
    """
    Main variation generator orchestrating all transformers.

    Usage:
        generator = VariationGenerator()
        variations = generator.get_representative_variations(template)
    """

    def __init__(self, config: Optional[VariationConfig] = None):
        """
        Initialize variation generator.

        Args:
            config: Optional configuration
        """
        self.config = config or VariationConfig()
        self._init_transformers()

    def _init_transformers(self):
        """Initialize all transformation modules."""
        from .transformers import (
            KeyboardMapper,
            Transliterator,
            SynonymExpander,
            FormalityShifter,
            TypoGenerator,
            NumberHandler,
            QuestionFormer,
            LanguageMixer,
        )

        self.keyboard_mapper = KeyboardMapper()
        self.transliterator = Transliterator()
        self.synonym_expander = SynonymExpander()
        self.formality_shifter = FormalityShifter()
        self.typo_generator = TypoGenerator()
        self.number_handler = NumberHandler()
        self.question_former = QuestionFormer()
        self.language_mixer = LanguageMixer()

    def generate_all_variations(
        self,
        template: Dict[str, Any]
    ) -> Iterator[QueryVariation]:
        """
        Generate all possible variations for a template.

        Uses iterator pattern for memory efficiency.

        Args:
            template: Query template dict

        Yields:
            QueryVariation objects
        """
        example_id = template.get("id", "unknown")
        question_uk = template.get("question_uk", "")
        question_en = template.get("question_en", "")

        variation_counter = 0

        # Original variations
        if question_uk:
            yield QueryVariation(
                original_id=example_id,
                variation_id=f"{example_id}_original_uk",
                text=question_uk,
                language="uk",
                transformation_types=[],
                is_representative=True,
            )
            variation_counter += 1

        if question_en:
            yield QueryVariation(
                original_id=example_id,
                variation_id=f"{example_id}_original_en",
                text=question_en,
                language="en",
                transformation_types=[],
                is_representative=True,
            )
            variation_counter += 1

        # Generate Ukrainian variations
        if question_uk:
            yield from self._generate_language_variations(
                text=question_uk,
                example_id=example_id,
                language="uk",
                counter_start=variation_counter
            )

        # Generate English variations
        if question_en:
            yield from self._generate_language_variations(
                text=question_en,
                example_id=example_id,
                language="en",
                counter_start=variation_counter + 50
            )

        # Mixed language variations
        if question_uk and question_en and self.config.include_language_mix:
            yield from self._generate_mixed_variations(
                text_uk=question_uk,
                text_en=question_en,
                example_id=example_id
            )

    def _generate_language_variations(
        self,
        text: str,
        example_id: str,
        language: str,
        counter_start: int = 0
    ) -> Iterator[QueryVariation]:
        """Generate variations for a single language."""
        counter = counter_start

        # Keyboard layout errors (Priority 1)
        if self.config.include_keyboard_errors:
            for var_text in self.keyboard_mapper.transform(text, language):
                yield QueryVariation(
                    original_id=example_id,
                    variation_id=f"{example_id}_keyboard_{counter}",
                    text=var_text,
                    language=language,
                    transformation_types=[TransformationType.KEYBOARD_LAYOUT],
                )
                counter += 1

        # Transliteration (Priority 1)
        if self.config.include_transliteration and language == "uk":
            for var_text in self.transliterator.transform(text, language)[:3]:
                yield QueryVariation(
                    original_id=example_id,
                    variation_id=f"{example_id}_translit_{counter}",
                    text=var_text,
                    language="latin",  # Transliterated
                    transformation_types=[TransformationType.TRANSLITERATION],
                )
                counter += 1

        # Synonyms (Priority 1)
        if self.config.include_synonyms:
            for var_text in self.synonym_expander.transform(text, language)[:3]:
                yield QueryVariation(
                    original_id=example_id,
                    variation_id=f"{example_id}_synonym_{counter}",
                    text=var_text,
                    language=language,
                    transformation_types=[TransformationType.SYNONYM],
                )
                counter += 1

        # Formality (Priority 2)
        if self.config.include_formality and language == "uk":
            for var_text in self.formality_shifter.transform(text, language)[:2]:
                yield QueryVariation(
                    original_id=example_id,
                    variation_id=f"{example_id}_formality_{counter}",
                    text=var_text,
                    language=language,
                    transformation_types=[TransformationType.FORMALITY],
                )
                counter += 1

        # Typos (Priority 2)
        if self.config.include_typos:
            for var_text in self.typo_generator.transform(text, language)[:2]:
                yield QueryVariation(
                    original_id=example_id,
                    variation_id=f"{example_id}_typo_{counter}",
                    text=var_text,
                    language=language,
                    transformation_types=[TransformationType.TYPO],
                )
                counter += 1

        # Numbers (Priority 2)
        if self.config.include_numbers:
            for var_text in self.number_handler.transform(text, language)[:2]:
                yield QueryVariation(
                    original_id=example_id,
                    variation_id=f"{example_id}_number_{counter}",
                    text=var_text,
                    language=language,
                    transformation_types=[TransformationType.NUMBER],
                )
                counter += 1

        # Question forms (Priority 3)
        if self.config.include_question_forms:
            for var_text in self.question_former.transform(text, language)[:2]:
                yield QueryVariation(
                    original_id=example_id,
                    variation_id=f"{example_id}_qform_{counter}",
                    text=var_text,
                    language=language,
                    transformation_types=[TransformationType.QUESTION_FORM],
                )
                counter += 1

    def _generate_mixed_variations(
        self,
        text_uk: str,
        text_en: str,
        example_id: str
    ) -> Iterator[QueryVariation]:
        """Generate mixed language variations."""
        counter = 0

        # Mix Ukrainian with English elements
        for var_text in self.language_mixer.transform(text_uk, "uk")[:2]:
            yield QueryVariation(
                original_id=example_id,
                variation_id=f"{example_id}_mixed_{counter}",
                text=var_text,
                language="mixed",
                transformation_types=[TransformationType.LANGUAGE_MIX],
            )
            counter += 1

        # Mix English with Ukrainian elements
        for var_text in self.language_mixer.transform(text_en, "en")[:2]:
            yield QueryVariation(
                original_id=example_id,
                variation_id=f"{example_id}_mixed_{counter}",
                text=var_text,
                language="mixed",
                transformation_types=[TransformationType.LANGUAGE_MIX],
            )
            counter += 1

    def get_representative_variations(
        self,
        template: Dict[str, Any]
    ) -> List[QueryVariation]:
        """
        Get a representative subset of variations for embedding.

        Strategy: Sample from each transformation type to ensure coverage.

        Args:
            template: Query template dict

        Returns:
            List of representative variations (~15 per template)
        """
        all_variations = list(self.generate_all_variations(template))

        # Group by transformation type
        by_type: Dict[str, List[QueryVariation]] = {}
        for var in all_variations:
            key = tuple(sorted(t.value for t in var.transformation_types)) or ('original',)
            str_key = str(key)
            if str_key not in by_type:
                by_type[str_key] = []
            by_type[str_key].append(var)

        # Sample from each group
        representative = []
        samples_per_type = max(
            1,
            self.config.representative_sample_size // max(len(by_type), 1)
        )

        for type_key, variations in by_type.items():
            sampled = random.sample(
                variations,
                min(samples_per_type, len(variations))
            )
            for var in sampled:
                var.is_representative = True
                representative.append(var)

        # Ensure we always have originals
        originals = [v for v in all_variations if not v.transformation_types]
        for orig in originals:
            if orig not in representative:
                orig.is_representative = True
                representative.insert(0, orig)

        return representative[:self.config.representative_sample_size]

    def correct_query(self, query: str) -> List[str]:
        """
        Attempt to correct a query that might have errors.

        Useful at query time to detect and fix keyboard errors,
        transliteration, etc.

        Args:
            query: User input query

        Returns:
            List of possible corrected queries
        """
        corrections = [query]  # Always include original

        # Check for keyboard layout error
        keyboard_correction = self.keyboard_mapper.correct_layout_error(query)
        if keyboard_correction:
            corrections.append(keyboard_correction)

        # Check if transliterated
        if self.transliterator.detect_transliteration(query):
            ukrainian = self.transliterator.transliterate_to_ukrainian(query)
            if ukrainian:
                corrections.append(ukrainian)

        return list(set(corrections))

    def get_stats(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistics about variations for a template.

        Args:
            template: Query template dict

        Returns:
            Statistics dict
        """
        all_vars = list(self.generate_all_variations(template))
        representative = self.get_representative_variations(template)

        # Count by type
        type_counts: Dict[str, int] = {}
        for var in all_vars:
            for t in var.transformation_types:
                type_counts[t.value] = type_counts.get(t.value, 0) + 1
        if not any(v.transformation_types for v in all_vars):
            type_counts['original'] = len([v for v in all_vars if not v.transformation_types])

        return {
            "total_variations": len(all_vars),
            "representative_count": len(representative),
            "by_type": type_counts,
            "languages": {
                "uk": len([v for v in all_vars if v.language == "uk"]),
                "en": len([v for v in all_vars if v.language == "en"]),
                "mixed": len([v for v in all_vars if v.language == "mixed"]),
                "latin": len([v for v in all_vars if v.language == "latin"]),
            }
        }


# Convenience function
def create_generator(
    include_typos: bool = True,
    include_keyboard_errors: bool = True,
    representative_size: int = 15
) -> VariationGenerator:
    """Create a configured variation generator."""
    config = VariationConfig(
        include_typos=include_typos,
        include_keyboard_errors=include_keyboard_errors,
        representative_sample_size=representative_size,
    )
    return VariationGenerator(config)


if __name__ == "__main__":
    # Test the generator
    generator = VariationGenerator()

    test_template = {
        "id": "test_001",
        "question_uk": "Покажи топ 10 товарів за продажами",
        "question_en": "Show top 10 products by sales",
    }

    print("=" * 60)
    print("VARIATION GENERATOR TEST")
    print("=" * 60)

    stats = generator.get_stats(test_template)
    print(f"\nTotal variations: {stats['total_variations']}")
    print(f"Representative: {stats['representative_count']}")
    print(f"\nBy type:")
    for t, count in stats['by_type'].items():
        print(f"  {t}: {count}")

    print(f"\nBy language:")
    for lang, count in stats['languages'].items():
        print(f"  {lang}: {count}")

    print("\n" + "=" * 60)
    print("REPRESENTATIVE VARIATIONS:")
    print("=" * 60)

    for var in generator.get_representative_variations(test_template):
        types = [t.value for t in var.transformation_types] or ['original']
        print(f"\n[{var.language}] {', '.join(types)}")
        print(f"  {var.text}")
