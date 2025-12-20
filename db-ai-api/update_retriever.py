# -*- coding: utf-8 -*-
"""Script to update retriever.py with all 8 transformers."""

# Read the file
with open('training_data/retriever.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_correct = '''    def correct_query(self, query: str) -> List[str]:
        """
        Attempt to correct a query that might have errors.

        Detects and fixes:
        - Keyboard layout errors (Ukrainian typed with English keyboard)
        - Transliterated input (Latin spelling of Ukrainian words)
        - Synonym variants (alternative business terms)
        - Number format normalization (words to digits)

        Args:
            query: User input query

        Returns:
            List of corrected query variants (including original)
        """
        corrections = [query]

        if not self.enable_query_correction:
            return corrections

        # 1. Check for keyboard layout error (highest priority - common mistake)
        if self.keyboard_mapper:
            layout_error = self.keyboard_mapper.detect_layout_error(query)
            if layout_error:
                corrected = self.keyboard_mapper.correct_layout_error(query)
                if corrected and corrected != query:
                    corrections.append(corrected)
                    logger.debug(f"Keyboard correction: '{query}' -> '{corrected}'")

        # 2. Check for transliteration (Latin to Cyrillic)
        if self.transliterator:
            if self.transliterator.detect_transliteration(query):
                ukrainian = self.transliterator.transliterate_to_ukrainian(query)
                if ukrainian and ukrainian != query:
                    corrections.append(ukrainian)
                    logger.debug(f"Transliteration: '{query}' -> '{ukrainian}'")

        # 3. Expand synonyms (add alternative business terms)
        if self.synonym_expander:
            if self.synonym_expander.can_transform(query, "uk") or self.synonym_expander.can_transform(query, "en"):
                # Detect language
                has_cyrillic = any(ord(c) > 1024 and ord(c) < 1280 for c in query)
                lang = "uk" if has_cyrillic else "en"
                synonym_variants = self.synonym_expander.transform(query, lang)
                # Add top 2 synonym variants
                for variant in synonym_variants[:2]:
                    if variant and variant != query and variant not in corrections:
                        corrections.append(variant)
                        logger.debug(f"Synonym variant: '{query}' -> '{variant}'")

        # 4. Normalize numbers (words to digits for better matching)
        if self.number_handler:
            if self.number_handler.can_transform(query, "uk") or self.number_handler.can_transform(query, "en"):
                has_cyrillic = any(ord(c) > 1024 and ord(c) < 1280 for c in query)
                lang = "uk" if has_cyrillic else "en"
                # Convert word numbers to digits
                digit_variants = self.number_handler.words_to_digits(query, lang)
                for variant in digit_variants[:1]:  # Just one normalization
                    if variant and variant != query and variant not in corrections:
                        corrections.append(variant)
                        logger.debug(f"Number normalization: '{query}' -> '{variant}'")

        # Limit to 5 variants max
        return list(set(corrections))[:5]'''

new_correct = '''    def correct_query(self, query: str) -> List[str]:
        """
        Attempt to correct a query using ALL 8 transformers.

        Detects and fixes:
        1. Keyboard layout errors (Ukrainian typed with English keyboard)
        2. Transliterated input (Latin spelling of Ukrainian words)
        3. Synonym variants (alternative business terms)
        4. Number format normalization (words to digits)
        5. Formality variants (casual/formal)
        6. Typo detection and correction
        7. Question form variants
        8. Language mixing (UK+EN)

        Args:
            query: User input query

        Returns:
            List of corrected query variants (including original)
        """
        corrections = [query]

        if not self.enable_query_correction:
            return corrections

        # Detect language
        has_cyrillic = any(ord(c) > 1024 and ord(c) < 1280 for c in query)
        lang = "uk" if has_cyrillic else "en"

        # 1. Check for keyboard layout error (highest priority - common mistake)
        if self.keyboard_mapper:
            layout_error = self.keyboard_mapper.detect_layout_error(query)
            if layout_error:
                corrected = self.keyboard_mapper.correct_layout_error(query)
                if corrected and corrected != query:
                    corrections.append(corrected)
                    logger.debug(f"Keyboard correction: '{query}' -> '{corrected}'")

        # 2. Check for transliteration (Latin to Cyrillic)
        if self.transliterator:
            if self.transliterator.detect_transliteration(query):
                ukrainian = self.transliterator.transliterate_to_ukrainian(query)
                if ukrainian and ukrainian != query:
                    corrections.append(ukrainian)
                    logger.debug(f"Transliteration: '{query}' -> '{ukrainian}'")

        # 3. Expand synonyms (add alternative business terms)
        if self.synonym_expander:
            if self.synonym_expander.can_transform(query, lang):
                synonym_variants = self.synonym_expander.transform(query, lang)
                for variant in synonym_variants[:2]:
                    if variant and variant != query and variant not in corrections:
                        corrections.append(variant)
                        logger.debug(f"Synonym variant: '{query}' -> '{variant}'")

        # 4. Normalize numbers (words to digits for better matching)
        if self.number_handler:
            if self.number_handler.can_transform(query, lang):
                digit_variants = self.number_handler.words_to_digits(query, lang)
                for variant in digit_variants[:1]:
                    if variant and variant != query and variant not in corrections:
                        corrections.append(variant)
                        logger.debug(f"Number normalization: '{query}' -> '{variant}'")

        # 5. Formality variants (casual <-> formal)
        if self.formality_shifter:
            if self.formality_shifter.can_transform(query, lang):
                formality_variants = self.formality_shifter.transform(query, lang)
                for variant in formality_variants[:1]:
                    if variant and variant != query and variant not in corrections:
                        corrections.append(variant)
                        logger.debug(f"Formality variant: '{query}' -> '{variant}'")

        # 6. Typo correction
        if self.typo_generator:
            if self.typo_generator.can_transform(query, lang):
                typo_variants = self.typo_generator.transform(query, lang)
                for variant in typo_variants[:1]:
                    if variant and variant != query and variant not in corrections:
                        corrections.append(variant)
                        logger.debug(f"Typo correction: '{query}' -> '{variant}'")

        # 7. Question form variants
        if self.question_former:
            if self.question_former.can_transform(query, lang):
                question_variants = self.question_former.transform(query, lang)
                for variant in question_variants[:1]:
                    if variant and variant != query and variant not in corrections:
                        corrections.append(variant)
                        logger.debug(f"Question form: '{query}' -> '{variant}'")

        # 8. Language mixing (for bilingual queries)
        if self.language_mixer:
            if self.language_mixer.can_transform(query, lang):
                mixed_variants = self.language_mixer.transform(query, lang)
                for variant in mixed_variants[:1]:
                    if variant and variant != query and variant not in corrections:
                        corrections.append(variant)
                        logger.debug(f"Language mix: '{query}' -> '{variant}'")

        # Limit to 8 variants max (original + 7 corrections)
        return list(set(corrections))[:8]'''

if old_correct in content:
    content = content.replace(old_correct, new_correct)
    with open('training_data/retriever.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully updated correct_query with all 8 transformers")
else:
    print("Old correct_query not found - may already be updated")
