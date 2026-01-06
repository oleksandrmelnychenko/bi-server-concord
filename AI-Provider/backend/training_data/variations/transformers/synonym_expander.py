"""Synonym expansion for Ukrainian business terms."""
from typing import List, Dict, Set, Tuple
from pathlib import Path
from functools import lru_cache
import json
from .base import BaseTransformer


# Module-level caching for synonym expansion
@lru_cache(maxsize=1024)
def _cached_can_transform(text: str, language: str) -> bool:
    """Cached check if text contains words with known synonyms."""
    synonyms = _UK_SYNONYMS if language == "uk" else _EN_SYNONYMS
    text_lower = text.lower()
    return any(word in text_lower for word in synonyms)


@lru_cache(maxsize=512)
def _cached_expand_synonyms(text: str, language: str, max_variations: int = 5) -> Tuple[str, ...]:
    """Cached synonym expansion using default dictionaries."""
    synonyms = _UK_SYNONYMS if language == "uk" else _EN_SYNONYMS

    text_lower = text.lower()
    words = text_lower.split()
    variations: Set[str] = set()

    # Find which words have synonyms
    expandable_positions = []
    for i, word in enumerate(words):
        word_base = word.rstrip(',.?!')
        if word_base in synonyms:
            expandable_positions.append((i, word_base, word[len(word_base):]))

    if not expandable_positions:
        return tuple()

    # Generate variations by replacing one word at a time
    for pos, word_base, suffix in expandable_positions:
        for synonym in synonyms[word_base][:3]:
            new_words = words.copy()
            new_words[pos] = synonym + suffix
            variation = ' '.join(new_words)
            if variation != text_lower:
                variations.add(variation)
                if len(variations) >= max_variations:
                    return tuple(variations)

    # Generate combinations (2 words replaced)
    if len(expandable_positions) >= 2:
        for i, (pos1, word1, suf1) in enumerate(expandable_positions):
            for pos2, word2, suf2 in expandable_positions[i + 1:]:
                for syn1 in synonyms[word1][:2]:
                    for syn2 in synonyms[word2][:2]:
                        new_words = words.copy()
                        new_words[pos1] = syn1 + suf1
                        new_words[pos2] = syn2 + suf2
                        variation = ' '.join(new_words)
                        if variation != text_lower:
                            variations.add(variation)
                            if len(variations) >= max_variations:
                                return tuple(variations)

    return tuple(variations)


# Module-level synonym dictionaries for cached functions
_UK_SYNONYMS = {
    'товари': ['продукти', 'продукція', 'вироби', 'позиції', 'артикули'],
    'товарів': ['продуктів', 'продукції', 'виробів', 'позицій'],
    'товар': ['продукт', 'виріб', 'позиція', 'артикул'],
    'товару': ['продукту', 'виробу', 'позиції'],
    'клієнти': ['покупці', 'замовники', 'контрагенти', 'споживачі', 'партнери'],
    'клієнтів': ['покупців', 'замовників', 'контрагентів', 'споживачів'],
    'клієнт': ['покупець', 'замовник', 'контрагент', 'споживач'],
    'клієнта': ['покупця', 'замовника', 'контрагента'],
    'продажі': ['реалізація', 'збут', 'торгівля', 'оборот'],
    'продажів': ['реалізації', 'збуту', 'торгівлі'],
    'продаж': ['реалізація', 'збут'],
    'замовлення': ['заявки', 'ордери', 'закази', 'накладні'],
    'замовлень': ['заявок', 'ордерів', 'заказів'],
    'борги': ['заборгованість', 'дебіторка', 'боргові зобовязання', 'несплата'],
    'боргів': ['заборгованості', 'дебіторки'],
    'борг': ['заборгованість', 'дебіторка'],
    'виручка': ['дохід', 'прибуток', 'оборот', 'надходження', 'виторг'],
    'виручки': ['доходу', 'прибутку', 'обороту', 'надходжень'],
    'платежі': ['оплати', 'перекази', 'транзакції', 'надходження'],
    'платежів': ['оплат', 'переказів', 'транзакцій'],
    'склад': ['сховище', 'магазин', 'база'],
    'складі': ['сховищі', 'магазині', 'базі'],
    'залишки': ['запаси', 'наявність', 'рештки', 'стоки'],
    'залишків': ['запасів', 'наявності'],
    'місяць': ['період', 'термін'],
    'місяці': ['періоди'],
    'рік': ['період'],
    'року': ['періоду'],
    'покажи': ['виведи', 'дай', 'відобрази', 'надай', 'скинь'],
    'показати': ['вивести', 'отримати', 'відобразити', 'надати'],
    'знайди': ['відшукай', 'пошук', 'шукай'],
    'знайти': ['відшукати', 'пошук'],
    'порахуй': ['підрахуй', 'полічи', 'обчисли'],
    'порахувати': ['підрахувати', 'полічити', 'обчислити'],
    'скільки': ['яка кількість', 'який обсяг', 'кількість'],
    'найбільше': ['найбільш', 'максимум', 'більше всього'],
    'найменше': ['найменш', 'мінімум', 'менше всього'],
    'топ': ['найкращі', 'перші', 'найпопулярніші', 'лідери'],
    'найкращі': ['топ', 'перші', 'кращі', 'лідери'],
    'активні': ['діючі', 'робочі', 'поточні'],
    'неактивні': ['недіючі', 'застарілі', 'архівні'],
}

_EN_SYNONYMS = {
    'products': ['items', 'goods', 'merchandise', 'articles'],
    'product': ['item', 'good', 'article'],
    'customers': ['clients', 'buyers', 'purchasers'],
    'customer': ['client', 'buyer', 'purchaser'],
    'sales': ['revenue', 'turnover', 'transactions'],
    'sale': ['transaction', 'deal'],
    'orders': ['purchases', 'requests'],
    'order': ['purchase', 'request'],
    'debts': ['receivables', 'amounts due', 'outstanding'],
    'debt': ['receivable', 'amount due'],
    'revenue': ['income', 'earnings', 'turnover'],
    'payments': ['transactions', 'transfers'],
    'stock': ['inventory', 'supply', 'reserves'],
    'warehouse': ['depot', 'storage'],
    'show': ['display', 'list', 'get', 'retrieve'],
    'find': ['search', 'locate', 'look for'],
    'count': ['total', 'calculate', 'sum up'],
    'top': ['best', 'leading', 'first'],
    'best': ['top', 'leading', 'highest'],
    'active': ['current', 'ongoing'],
    'inactive': ['dormant', 'archived'],
}


class SynonymExpander(BaseTransformer):
    """
    Expand business terms to their synonyms.

    Focuses on Ukrainian e-commerce/business vocabulary commonly used
    in the ConcordDb database context.
    """

    # Core Ukrainian business synonyms (embedded for reliability)
    UKRAINIAN_SYNONYMS = {
        # Products
        'товари': ['продукти', 'продукція', 'вироби', 'позиції', 'артикули'],
        'товарів': ['продуктів', 'продукції', 'виробів', 'позицій'],
        'товар': ['продукт', 'виріб', 'позиція', 'артикул'],
        'товару': ['продукту', 'виробу', 'позиції'],

        # Customers
        'клієнти': ['покупці', 'замовники', 'контрагенти', 'споживачі', 'партнери'],
        'клієнтів': ['покупців', 'замовників', 'контрагентів', 'споживачів'],
        'клієнт': ['покупець', 'замовник', 'контрагент', 'споживач'],
        'клієнта': ['покупця', 'замовника', 'контрагента'],

        # Sales
        'продажі': ['реалізація', 'збут', 'торгівля', 'оборот'],
        'продажів': ['реалізації', 'збуту', 'торгівлі'],
        'продаж': ['реалізація', 'збут'],

        # Orders
        'замовлення': ['заявки', 'ордери', 'закази', 'накладні'],
        'замовлень': ['заявок', 'ордерів', 'заказів'],

        # Financial
        'борги': ['заборгованість', 'дебіторка', 'боргові зобовязання', 'несплата'],
        'боргів': ['заборгованості', 'дебіторки'],
        'борг': ['заборгованість', 'дебіторка'],

        'виручка': ['дохід', 'прибуток', 'оборот', 'надходження', 'виторг'],
        'виручки': ['доходу', 'прибутку', 'обороту', 'надходжень'],

        'платежі': ['оплати', 'перекази', 'транзакції', 'надходження'],
        'платежів': ['оплат', 'переказів', 'транзакцій'],

        # Inventory
        'склад': ['сховище', 'магазин', 'база'],
        'складі': ['сховищі', 'магазині', 'базі'],
        'залишки': ['запаси', 'наявність', 'рештки', 'стоки'],
        'залишків': ['запасів', 'наявності'],

        # Time periods
        'місяць': ['період', 'термін'],
        'місяці': ['періоди'],
        'рік': ['період'],
        'року': ['періоду'],

        # Actions
        'покажи': ['виведи', 'дай', 'відобрази', 'надай', 'скинь'],
        'показати': ['вивести', 'отримати', 'відобразити', 'надати'],
        'знайди': ['відшукай', 'пошук', 'шукай'],
        'знайти': ['відшукати', 'пошук'],
        'порахуй': ['підрахуй', 'полічи', 'обчисли'],
        'порахувати': ['підрахувати', 'полічити', 'обчислити'],

        # Quantities
        'скільки': ['яка кількість', 'який обсяг', 'кількість'],
        'найбільше': ['найбільш', 'максимум', 'більше всього'],
        'найменше': ['найменш', 'мінімум', 'менше всього'],

        # Adjectives
        'топ': ['найкращі', 'перші', 'найпопулярніші', 'лідери'],
        'найкращі': ['топ', 'перші', 'кращі', 'лідери'],
        'активні': ['діючі', 'робочі', 'поточні'],
        'неактивні': ['недіючі', 'застарілі', 'архівні'],
    }

    # English synonyms for bilingual support
    ENGLISH_SYNONYMS = {
        # Products
        'products': ['items', 'goods', 'merchandise', 'articles'],
        'product': ['item', 'good', 'article'],

        # Customers
        'customers': ['clients', 'buyers', 'purchasers'],
        'customer': ['client', 'buyer', 'purchaser'],

        # Sales
        'sales': ['revenue', 'turnover', 'transactions'],
        'sale': ['transaction', 'deal'],

        # Orders
        'orders': ['purchases', 'requests'],
        'order': ['purchase', 'request'],

        # Financial
        'debts': ['receivables', 'amounts due', 'outstanding'],
        'debt': ['receivable', 'amount due'],
        'revenue': ['income', 'earnings', 'turnover'],
        'payments': ['transactions', 'transfers'],

        # Inventory
        'stock': ['inventory', 'supply', 'reserves'],
        'warehouse': ['depot', 'storage'],

        # Actions
        'show': ['display', 'list', 'get', 'retrieve'],
        'find': ['search', 'locate', 'look for'],
        'count': ['total', 'calculate', 'sum up'],

        # Adjectives
        'top': ['best', 'leading', 'first'],
        'best': ['top', 'leading', 'highest'],
        'active': ['current', 'ongoing'],
        'inactive': ['dormant', 'archived'],
    }

    def __init__(self, dictionary_path: str = None):
        """
        Initialize synonym expander.

        Args:
            dictionary_path: Optional path to JSON dictionary file
        """
        self.uk_synonyms = dict(self.UKRAINIAN_SYNONYMS)
        self.en_synonyms = dict(self.ENGLISH_SYNONYMS)

        # Load additional synonyms from file if provided
        if dictionary_path:
            self._load_dictionary(dictionary_path)

    @property
    def name(self) -> str:
        return "synonym_expander"

    def transform(self, text: str, language: str = "uk") -> List[str]:
        """
        Generate synonym variations of the text (cached for default dictionaries).

        Args:
            text: Input text
            language: Language code

        Returns:
            List of synonym variations (not including original)
        """
        # Use cached function for default dictionaries (common case)
        if not hasattr(self, '_custom_dict_loaded') or not self._custom_dict_loaded:
            return list(_cached_expand_synonyms(text, language))

        # Fall back to instance method if custom dictionary loaded
        synonyms = self.uk_synonyms if language == "uk" else self.en_synonyms
        return self._expand_synonyms(text, synonyms)

    def _expand_synonyms(
        self,
        text: str,
        synonyms: Dict[str, List[str]],
        max_variations: int = 5
    ) -> List[str]:
        """
        Expand text by replacing words with synonyms.

        Args:
            text: Input text
            synonyms: Synonym dictionary
            max_variations: Maximum variations to generate

        Returns:
            List of expanded variations
        """
        text_lower = text.lower()
        words = text_lower.split()
        variations: Set[str] = set()

        # Find which words have synonyms
        expandable_positions = []
        for i, word in enumerate(words):
            # Check word and common suffixed forms
            word_base = word.rstrip(',.?!')
            if word_base in synonyms:
                expandable_positions.append((i, word_base, word[len(word_base):]))

        if not expandable_positions:
            return []

        # Generate variations by replacing one word at a time
        for pos, word_base, suffix in expandable_positions:
            for synonym in synonyms[word_base][:3]:  # Limit synonyms per word
                new_words = words.copy()
                new_words[pos] = synonym + suffix
                variation = ' '.join(new_words)
                if variation != text_lower:
                    variations.add(variation)
                    if len(variations) >= max_variations:
                        return list(variations)

        # Generate combinations (2 words replaced)
        if len(expandable_positions) >= 2 and len(variations) < max_variations:
            for i, (pos1, word1, suf1) in enumerate(expandable_positions):
                for pos2, word2, suf2 in expandable_positions[i + 1:]:
                    for syn1 in synonyms[word1][:2]:
                        for syn2 in synonyms[word2][:2]:
                            new_words = words.copy()
                            new_words[pos1] = syn1 + suf1
                            new_words[pos2] = syn2 + suf2
                            variation = ' '.join(new_words)
                            if variation != text_lower:
                                variations.add(variation)
                                if len(variations) >= max_variations:
                                    return list(variations)

        return list(variations)

    def get_synonyms(self, word: str, language: str = "uk") -> List[str]:
        """
        Get synonyms for a specific word.

        Args:
            word: Word to find synonyms for
            language: Language code

        Returns:
            List of synonyms
        """
        synonyms = self.uk_synonyms if language == "uk" else self.en_synonyms
        return synonyms.get(word.lower(), [])

    def _load_dictionary(self, path: str) -> None:
        """Load additional synonyms from JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'uk' in data:
                self.uk_synonyms.update(data['uk'])
                self._custom_dict_loaded = True
            if 'en' in data:
                self.en_synonyms.update(data['en'])
                self._custom_dict_loaded = True
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def can_transform(self, text: str, language: str = "uk") -> bool:
        """Check if text contains words with known synonyms (cached)."""
        # Use cached function for default dictionaries (common case)
        if not hasattr(self, '_custom_dict_loaded') or not self._custom_dict_loaded:
            return _cached_can_transform(text, language)

        # Fall back to instance check if custom dictionary loaded
        synonyms = self.uk_synonyms if language == "uk" else self.en_synonyms
        text_lower = text.lower()
        return any(word in text_lower for word in synonyms)
