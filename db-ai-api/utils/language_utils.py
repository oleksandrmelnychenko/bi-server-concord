"""Ukrainian language support utilities."""
from typing import Optional, Dict, Any
import re


def detect_language(text: str) -> str:
    """
    Detect if text is Ukrainian or English.

    Args:
        text: Input text

    Returns:
        'uk' for Ukrainian, 'en' for English
    """
    try:
        from langdetect import detect
        lang = detect(text)
        return 'uk' if lang == 'uk' or lang == 'ru' else 'en'
    except:
        # Fallback: check for Cyrillic characters
        cyrillic_pattern = re.compile('[а-яА-ЯіїєґІЇЄҐ]')
        if cyrillic_pattern.search(text):
            return 'uk'
        return 'en'


def has_ukrainian(text: str) -> bool:
    """Check if text contains Ukrainian/Cyrillic characters."""
    cyrillic_pattern = re.compile('[а-яА-ЯіїєґІЇЄҐ]')
    return bool(cyrillic_pattern.search(text))


# Ukrainian → English table translations
TABLE_TRANSLATIONS_UK_EN = {
    "Client": "Клієнт",
    "Order": "Замовлення",
    "Product": "Продукт",
    "Sale": "Продаж",
    "User": "Користувач",
    "Organization": "Організація",
    "ClientWorkplace": "Робоче місце клієнта",
    "Payment": "Платіж",
    "Invoice": "Рахунок",
    "Agreement": "Угода",
    "Pricing": "Ціноутворення",
    "ProductPricing": "Ціни продуктів",
    "Supplier": "Постачальник",
    "Delivery": "Доставка",
    "Warehouse": "Склад",
}

# Reverse mapping
TABLE_TRANSLATIONS_EN_UK = {v: k for k, v in TABLE_TRANSLATIONS_UK_EN.items()}


# Column translations
COLUMN_TRANSLATIONS_UK_EN = {
    "ID": "Ідентифікатор",
    "Name": "Назва",
    "City": "Місто",
    "Address": "Адреса",
    "Phone": "Телефон",
    "Email": "Електронна пошта",
    "Created": "Створено",
    "Updated": "Оновлено",
    "Deleted": "Видалено",
    "Price": "Ціна",
    "Amount": "Сума",
    "Quantity": "Кількість",
    "Status": "Статус",
    "Description": "Опис",
    "Date": "Дата",
    "Total": "Загалом",
}

COLUMN_TRANSLATIONS_EN_UK = {v: k for k, v in COLUMN_TRANSLATIONS_UK_EN.items()}


# Common Ukrainian values and their variations
UKRAINIAN_VALUES = {
    "київ": ["Київ", "Kiev", "Kyiv", "Киев"],
    "львів": ["Львів", "Lviv", "Львов"],
    "одеса": ["Одеса", "Odesa", "Odessa", "Одесса"],
    "харків": ["Харків", "Kharkiv", "Харьков"],
    "дніпро": ["Дніпро", "Dnipro", "Днепр"],
    "запоріжжя": ["Запоріжжя", "Zaporizhzhia", "Запорожье"],
    "україна": ["Україна", "Ukraine", "Украина"],
    "польща": ["Польща", "Poland", "Polska", "Польша"],
    "німеччина": ["Німеччина", "Germany", "Deutschland", "Германия"],
}


def normalize_ukrainian_value(value: str) -> str:
    """
    Normalize Ukrainian city/country names to canonical form.

    Args:
        value: Input value

    Returns:
        Normalized canonical form
    """
    value_lower = value.lower().strip()

    for canonical, variations in UKRAINIAN_VALUES.items():
        if value_lower in [v.lower() for v in variations]:
            return variations[0]  # Return canonical Ukrainian form

    return value


def translate_table_to_ukrainian(table_name: str) -> str:
    """
    Translate English table name to Ukrainian.

    Args:
        table_name: English table name

    Returns:
        Ukrainian translation or original if not found
    """
    # Remove schema prefix if present
    clean_name = table_name.split('.')[-1]
    return TABLE_TRANSLATIONS_UK_EN.get(clean_name, clean_name)


def translate_column_to_ukrainian(column_name: str) -> str:
    """
    Translate English column name to Ukrainian.

    Args:
        column_name: English column name

    Returns:
        Ukrainian translation or original if not found
    """
    return COLUMN_TRANSLATIONS_UK_EN.get(column_name, column_name)


def format_number_ukrainian(number: float) -> str:
    """
    Format number in Ukrainian style (space as thousand separator).

    Args:
        number: Number to format

    Returns:
        Formatted string like "1 250 000"
    """
    return f"{number:,.0f}".replace(',', ' ')


def format_currency_ukrainian(amount: float, currency: str = "UAH") -> str:
    """
    Format currency in Ukrainian style.

    Args:
        amount: Amount
        currency: Currency code

    Returns:
        Formatted string like "1 250 000 UAH"
    """
    formatted = format_number_ukrainian(amount)
    return f"{formatted} {currency}"


def format_date_ukrainian(date_str: str) -> str:
    """
    Format date in Ukrainian style (DD.MM.YYYY).

    Args:
        date_str: ISO date string

    Returns:
        Ukrainian formatted date
    """
    from datetime import datetime

    try:
        if 'T' in date_str:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        else:
            dt = datetime.fromisoformat(date_str)
        return dt.strftime("%d.%m.%Y")
    except:
        return date_str


def create_ukrainian_document(table_name: str, row_data: Dict[str, Any],
                               primary_key: str = "ID") -> str:
    """
    Create rich Ukrainian language document from database row.

    Args:
        table_name: Table name
        row_data: Dictionary of column: value
        primary_key: Primary key column name

    Returns:
        Rich Ukrainian language document
    """
    table_uk = translate_table_to_ukrainian(table_name)
    doc_parts = []

    # Header
    doc_parts.append(f"=== Запис з таблиці {table_uk} ===")

    # Primary key
    if primary_key in row_data:
        doc_parts.append(f"ID запису: {row_data[primary_key]}")

    # Group columns by type
    basic_info = []
    financial_info = []
    dates = []
    other_info = []

    for column, value in row_data.items():
        if column == primary_key or value is None:
            continue

        column_uk = translate_column_to_ukrainian(column)

        # Handle different value types
        if isinstance(value, (int, float)) and column.lower() in ['price', 'amount', 'sum', 'total', 'cost']:
            formatted_value = format_currency_ukrainian(float(value))
            financial_info.append(f"- {column_uk}: {formatted_value}")
        elif 'date' in column.lower() or 'created' in column.lower() or 'updated' in column.lower():
            formatted_value = format_date_ukrainian(str(value))
            dates.append(f"- {column_uk}: {formatted_value}")
        elif column.lower() in ['name', 'title', 'city', 'address', 'phone', 'email']:
            basic_info.append(f"- {column_uk}: {value}")
        else:
            # Convert to string, limit length
            str_value = str(value)[:200]
            other_info.append(f"- {column_uk}: {str_value}")

    # Add sections
    if basic_info:
        doc_parts.append("\nОсновна інформація:")
        doc_parts.extend(basic_info[:10])  # Limit to 10

    if financial_info:
        doc_parts.append("\nФінансові дані:")
        doc_parts.extend(financial_info[:5])

    if dates:
        doc_parts.append("\nДати:")
        doc_parts.extend(dates[:5])

    if other_info:
        doc_parts.append("\nДодаткова інформація:")
        doc_parts.extend(other_info[:5])

    return "\n".join(doc_parts)


def extract_ukrainian_keywords(text: str) -> list[str]:
    """
    Extract Ukrainian keywords from text for better matching.

    Args:
        text: Input text

    Returns:
        List of keywords
    """
    # Remove punctuation
    cleaned = re.sub(r'[^\w\s]', ' ', text.lower())

    # Split and filter
    words = cleaned.split()

    # Remove common stop words
    stop_words = {'і', 'в', 'на', 'з', 'у', 'до', 'по', 'від', 'за', 'про', 'або', 'та', 'як', 'що', 'це', 'є', 'був', 'була', 'було', 'були'}

    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    return keywords
