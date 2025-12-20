"""Typo correction for user queries (reverses common typo patterns).

Unlike TypoGenerator (which creates typos for training data), this class
CORRECTS typos in user input by reversing common error patterns.
"""
from typing import List, Dict, Set
from .base import BaseTransformer


class TypoCorrector(BaseTransformer):
    """
    Detect and correct typos in user queries.

    Strategies:
    1. Fix doubled characters ("товвари" -> "товари")
    2. Fix transpositions ("товраи" -> "товари")
    3. Fix missing characters (compare to known words)
    4. Fix adjacent key errors (using keyboard maps)
    """

    # Common Ukrainian business words for correction reference
    UK_COMMON_WORDS: Set[str] = {
        # Products
        'товари', 'товар', 'товарів', 'товару', 'товаром',
        'продукти', 'продукт', 'продуктів', 'продукції', 'продукція',
        'вироби', 'виріб', 'виробів',
        'позиції', 'позиція', 'позицій',
        'артикули', 'артикул',
        # Customers
        'клієнти', 'клієнт', 'клієнтів', 'клієнта', 'клієнтом',
        'покупці', 'покупець', 'покупців',
        'замовники', 'замовник', 'замовників',
        'контрагенти', 'контрагент', 'контрагентів',
        # Sales
        'продажі', 'продаж', 'продажів', 'продажу',
        'реалізація', 'реалізації',
        'збут', 'збуту',
        # Orders
        'замовлення', 'замовлень', 'замовленню',
        'заявки', 'заявка', 'заявок',
        'ордери', 'ордер',
        # Financial
        'борги', 'борг', 'боргів', 'боргу',
        'заборгованість', 'заборгованості',
        'дебіторка', 'дебіторки',
        'платежі', 'платіж', 'платежів',
        'оплати', 'оплата', 'оплат',
        'виручка', 'виручки', 'виручку',
        'дохід', 'доходу', 'доходів',
        # Inventory
        'склад', 'складі', 'складу',
        'залишки', 'залишків', 'залишок',
        'запаси', 'запасів', 'запас',
        # Actions
        'покажи', 'показати', 'показує',
        'виведи', 'вивести', 'виводить',
        'знайди', 'знайти', 'знаходить',
        'пошук', 'пошуку',
        'порахуй', 'порахувати', 'підрахуй',
        'дай', 'дати',
        # Quantities
        'скільки', 'кількість', 'кількості',
        'всього', 'загалом', 'разом',
        # Adjectives
        'топ', 'найкращі', 'найкращих', 'перші', 'перших',
        'всі', 'всіх', 'активні', 'активних',
        'найбільше', 'найменше',
        # Time
        'рік', 'року', 'роки', 'років',
        'місяць', 'місяця', 'місяці', 'місяців',
        'день', 'дня', 'дні', 'днів',
        'тиждень', 'тижня', 'тижні',
        # Common
        'регіон', 'регіону', 'регіони', 'регіонів',
        'область', 'області', 'областей',
        'код', 'коду', 'коди', 'кодів',
        'назва', 'назви', 'назву',
    }

    # English common words
    EN_COMMON_WORDS: Set[str] = {
        'products', 'product', 'items', 'item',
        'customers', 'customer', 'clients', 'client',
        'sales', 'sale', 'orders', 'order',
        'debts', 'debt', 'payments', 'payment',
        'stock', 'inventory', 'warehouse',
        'show', 'list', 'find', 'search', 'count', 'get',
        'top', 'best', 'first', 'all', 'active',
        'year', 'month', 'day', 'week',
        'region', 'code', 'name', 'total',
    }

    # Adjacent keys on Ukrainian ЙЦУКЕН keyboard (for typo correction)
    UK_ADJACENT_KEYS: Dict[str, List[str]] = {
        'й': ['ц', 'ф', 'і'],
        'ц': ['й', 'у', 'ф', 'і', 'в'],
        'у': ['ц', 'к', 'і', 'в', 'а'],
        'к': ['у', 'е', 'в', 'а', 'п'],
        'е': ['к', 'н', 'а', 'п', 'р'],
        'н': ['е', 'г', 'п', 'р', 'о'],
        'г': ['н', 'ш', 'р', 'о', 'л'],
        'ш': ['г', 'щ', 'о', 'л', 'д'],
        'щ': ['ш', 'з', 'л', 'д', 'ж'],
        'з': ['щ', 'х', 'д', 'ж', 'є'],
        'х': ['з', 'ї', 'ж', 'є'],
        'ї': ['х', 'є'],
        'ф': ['й', 'ц', 'і', 'я', 'ч'],
        'і': ['ц', 'у', 'ф', 'в', 'я', 'ч', 'с'],
        'в': ['у', 'к', 'і', 'а', 'ч', 'с', 'м'],
        'а': ['к', 'е', 'в', 'п', 'с', 'м', 'и'],
        'п': ['е', 'н', 'а', 'р', 'м', 'и', 'т'],
        'р': ['н', 'г', 'п', 'о', 'и', 'т', 'ь'],
        'о': ['г', 'ш', 'р', 'л', 'т', 'ь', 'б'],
        'л': ['ш', 'щ', 'о', 'д', 'ь', 'б', 'ю'],
        'д': ['щ', 'з', 'л', 'ж', 'б', 'ю'],
        'ж': ['з', 'х', 'д', 'є', 'ю'],
        'є': ['х', 'ї', 'ж'],
        'я': ['ф', 'і', 'ч'],
        'ч': ['і', 'в', 'я', 'с'],
        'с': ['в', 'а', 'ч', 'м'],
        'м': ['а', 'п', 'с', 'и'],
        'и': ['п', 'р', 'м', 'т'],
        'т': ['р', 'о', 'и', 'ь'],
        'ь': ['о', 'л', 'т', 'б'],
        'б': ['л', 'д', 'ь', 'ю'],
        'ю': ['д', 'ж', 'б'],
    }

    @property
    def name(self) -> str:
        return "typo_corrector"

    def transform(self, text: str, language: str = "uk") -> List[str]:
        """Main transform method - returns corrected variants."""
        return self.correct_typos(text, language)

    def correct_typos(self, text: str, language: str = "uk") -> List[str]:
        """
        Attempt to correct typos in text.

        Returns list of possible corrections (may be empty if no typos detected).
        """
        corrections: Set[str] = set()
        words = text.lower().split()
        known_words = self.UK_COMMON_WORDS if language == "uk" else self.EN_COMMON_WORDS

        for i, word in enumerate(words):
            if len(word) < 3:
                continue

            # Skip if word is already known
            if word in known_words:
                continue

            # Try various correction strategies
            word_corrections: List[str] = []

            # 1. Fix doubled characters
            word_corrections.extend(self._fix_doubled_chars(word, known_words))

            # 2. Fix transpositions
            word_corrections.extend(self._fix_transpositions(word, known_words))

            # 3. Fix missing characters
            word_corrections.extend(self._fix_missing_chars(word, known_words))

            # 4. Fix extra characters
            word_corrections.extend(self._fix_extra_chars(word, known_words))

            # 5. Fix adjacent key errors
            if language == "uk":
                word_corrections.extend(self._fix_adjacent_key(word, known_words))

            # Generate corrected sentences
            for corrected_word in word_corrections:
                if corrected_word != word:
                    new_words = words.copy()
                    new_words[i] = corrected_word
                    corrections.add(' '.join(new_words))

        return list(corrections)[:5]  # Limit to 5 corrections

    def _fix_doubled_chars(self, word: str, known_words: Set[str]) -> List[str]:
        """Fix accidentally doubled characters: 'товвари' -> 'товари'"""
        results = []
        for i in range(len(word) - 1):
            if word[i] == word[i + 1]:
                fixed = word[:i] + word[i + 1:]
                if fixed in known_words:
                    results.append(fixed)
        return results

    def _fix_transpositions(self, word: str, known_words: Set[str]) -> List[str]:
        """Fix swapped adjacent chars: 'товраи' -> 'товари'"""
        results = []
        for i in range(len(word) - 1):
            # Swap chars at position i and i+1
            fixed = word[:i] + word[i + 1] + word[i] + word[i + 2:]
            if fixed in known_words:
                results.append(fixed)
        return results

    def _fix_missing_chars(self, word: str, known_words: Set[str]) -> List[str]:
        """Try adding missing characters based on known words."""
        results = []
        # Find similar known words that are 1 char longer
        for known in known_words:
            if len(known) == len(word) + 1:
                # Check if word could be known with 1 char removed
                for i in range(len(known)):
                    if known[:i] + known[i + 1:] == word:
                        results.append(known)
                        break
        return results[:3]  # Limit to prevent explosion

    def _fix_extra_chars(self, word: str, known_words: Set[str]) -> List[str]:
        """Try removing extra characters."""
        results = []
        # Find known words that are 1 char shorter
        for known in known_words:
            if len(known) == len(word) - 1:
                # Check if known is word with 1 char removed
                for i in range(len(word)):
                    if word[:i] + word[i + 1:] == known:
                        results.append(known)
                        break
        return results[:3]

    def _fix_adjacent_key(self, word: str, known_words: Set[str]) -> List[str]:
        """Fix adjacent key typos using keyboard map."""
        results = []

        # Try replacing each character with adjacent keys
        for i, char in enumerate(word):
            if char in self.UK_ADJACENT_KEYS:
                for adjacent in self.UK_ADJACENT_KEYS[char]:
                    fixed = word[:i] + adjacent + word[i + 1:]
                    if fixed in known_words:
                        results.append(fixed)
                        # Found a valid correction, move to next position
                        break

        return results[:3]

    def can_transform(self, text: str, language: str = "uk") -> bool:
        """Check if text might contain typos to correct."""
        words = text.lower().split()
        known_words = self.UK_COMMON_WORDS if language == "uk" else self.EN_COMMON_WORDS

        # Check if any word is NOT in known words (could be a typo)
        for word in words:
            if len(word) >= 3 and word not in known_words:
                # Check if it's close to any known word
                for known in known_words:
                    # Similar length suggests possible typo
                    if abs(len(known) - len(word)) <= 1:
                        return True
        return False

    def detect_likely_typo(self, word: str, language: str = "uk") -> bool:
        """Check if a specific word is likely a typo."""
        known_words = self.UK_COMMON_WORDS if language == "uk" else self.EN_COMMON_WORDS

        if word in known_words:
            return False

        # Check if fixing any typo type produces a known word
        if self._fix_doubled_chars(word, known_words):
            return True
        if self._fix_transpositions(word, known_words):
            return True
        if self._fix_missing_chars(word, known_words):
            return True
        if self._fix_extra_chars(word, known_words):
            return True
        if language == "uk" and self._fix_adjacent_key(word, known_words):
            return True

        return False
