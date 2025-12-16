#!/usr/bin/env python3
"""
Quick test script for Ukrainian language utilities
"""
from utils.language_utils import (
    detect_language,
    has_ukrainian,
    translate_table_to_ukrainian,
    translate_column_to_ukrainian,
    format_number_ukrainian,
    format_currency_ukrainian,
    format_date_ukrainian,
    normalize_ukrainian_value,
    extract_ukrainian_keywords,
    create_ukrainian_document
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def test_language_detection():
    """Test language detection."""
    console.print("\n[bold cyan]1. Language Detection[/bold cyan]\n")

    tests = [
        "Скільки клієнтів з Києва?",
        "How many clients in Kyiv?",
        "Покажи всі замовлення",
        "Show all orders"
    ]

    table = Table()
    table.add_column("Text", style="white")
    table.add_column("Language", style="green")
    table.add_column("Has Ukrainian", style="yellow")

    for text in tests:
        lang = detect_language(text)
        has_uk = has_ukrainian(text)
        table.add_row(text, lang, "✓" if has_uk else "✗")

    console.print(table)


def test_translations():
    """Test table/column translations."""
    console.print("\n[bold cyan]2. Table/Column Translations[/bold cyan]\n")

    table = Table()
    table.add_column("English", style="white")
    table.add_column("Ukrainian", style="green")

    # Tables
    for eng in ["Client", "Order", "Product", "Payment"]:
        ukr = translate_table_to_ukrainian(eng)
        table.add_row(eng, ukr)

    console.print(table)
    console.print()

    table2 = Table()
    table2.add_column("English", style="white")
    table2.add_column("Ukrainian", style="green")

    # Columns
    for eng in ["Name", "City", "Price", "Created"]:
        ukr = translate_column_to_ukrainian(eng)
        table2.add_row(eng, ukr)

    console.print(table2)


def test_formatting():
    """Test number/date formatting."""
    console.print("\n[bold cyan]3. Number/Currency/Date Formatting[/bold cyan]\n")

    table = Table()
    table.add_column("Type", style="cyan")
    table.add_column("Input", style="white")
    table.add_column("Output", style="green")

    table.add_row("Number", "1250000", format_number_ukrainian(1250000))
    table.add_row("Currency", "1250000", format_currency_ukrainian(1250000))
    table.add_row("Date", "2024-03-15", format_date_ukrainian("2024-03-15"))
    table.add_row("Date+Time", "2024-03-15T10:30:00", format_date_ukrainian("2024-03-15T10:30:00"))

    console.print(table)


def test_normalization():
    """Test value normalization."""
    console.print("\n[bold cyan]4. City/Country Normalization[/bold cyan]\n")

    table = Table()
    table.add_column("Input", style="white")
    table.add_column("Normalized", style="green")

    variations = [
        "Kiev", "Kyiv", "Киев",
        "Lviv", "Львов",
        "Ukraine", "Украина"
    ]

    for var in variations:
        normalized = normalize_ukrainian_value(var)
        table.add_row(var, normalized)

    console.print(table)


def test_keywords():
    """Test keyword extraction."""
    console.print("\n[bold cyan]5. Keyword Extraction[/bold cyan]\n")

    texts = [
        "Скільки клієнтів з Києва?",
        "Покажи всі замовлення за останній місяць",
        "Топ 10 найдорожчих товарів"
    ]

    for text in texts:
        keywords = extract_ukrainian_keywords(text)
        console.print(f"[white]{text}[/white]")
        console.print(f"[green]Keywords: {', '.join(keywords)}[/green]\n")


def test_document_creation():
    """Test Ukrainian document creation."""
    console.print("\n[bold cyan]6. Ukrainian Document Creation[/bold cyan]\n")

    # Sample row data
    row_data = {
        "ID": 123,
        "Name": "ТОВ \"Горизонт Технології\"",
        "City": "Київ",
        "Address": "вул. Хрещатик, 1",
        "Phone": "+380442345678",
        "Email": "info@horizont.ua",
        "Created": "2024-01-15",
        "Amount": 1250000
    }

    doc = create_ukrainian_document(
        table_name="Client",
        row_data=row_data,
        primary_key="ID"
    )

    console.print(Panel(
        doc,
        title="Generated Ukrainian Document",
        border_style="green"
    ))


def main():
    """Run all tests."""
    console.print("\n[bold white on blue] Ukrainian Language Utilities Test Suite [/bold white on blue]\n")

    test_language_detection()
    test_translations()
    test_formatting()
    test_normalization()
    test_keywords()
    test_document_creation()

    console.print("\n[bold green]✅ All tests completed![/bold green]\n")


if __name__ == "__main__":
    main()
