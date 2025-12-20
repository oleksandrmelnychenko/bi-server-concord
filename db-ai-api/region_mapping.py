"""
Ukrainian Region Code Mapping
Maps database region codes to full Ukrainian names
"""

# Ukrainian oblast codes to full names
REGION_CODES = {
    "CE": "Чернівецька область",
    "CK": "Черкаська область",
    "CN": "Чернігівська область",
    "DN": "Донецька область",
    "DP": "Дніпропетровська область",
    "GT": "Житомирська область",
    "IF": "Івано-Франківська область",
    "KD": "Краснодарський край",
    "KI": "Київ",
    "KR": "Кіровоградська область",
    "LK": "Луганська область",
    "LV": "Львівська область",
    "MD": "Молдова",
    "MI": "Миколаївська область",
    "OD": "Одеська область",
    "PA": "Полтавська область",
    "RI": "Рівненська область",
    "RS": "Росія",
    "SM": "Сумська область",
    "TE": "Тернопільська область",
    "VI": "Вінницька область",
    "VL": "Волинська область",
    "XM": "Хмельницька область",
    "XN": "Херсонська область",
    "XV": "Харківська область",
    "ZK": "Закарпатська область",
    "ZP": "Запорізька область",
    "BL": "Білорусь",
    # P-prefixed codes (likely partner/branch regions)
    "PCE": "Партнер Чернівецька",
    "PCK": "Партнер Черкаська",
    "PCN": "Партнер Чернігівська",
    "PDN": "Партнер Донецька",
    "PDP": "Партнер Дніпропетровська",
    "PGT": "Партнер Житомирська",
    "PIF": "Партнер Івано-Франківська",
    "PKD": "Партнер Краснодарський",
    "PKI": "Партнер Київ",
    "PKR": "Партнер Кіровоградська",
    "PL": "Польща",
    "PLK": "Партнер Луганська",
    "PLV": "Партнер Львівська",
    "PMD": "Партнер Молдова",
    "PMI": "Партнер Миколаївська",
    "POD": "Партнер Одеська",
    "PPA": "Партнер Полтавська",
    "PRI": "Партнер Рівненська",
    "PRS": "Партнер Росія",
    "PSM": "Партнер Сумська",
    "PTE": "Партнер Тернопільська",
    "PVI": "Партнер Вінницька",
    "PVL": "Партнер Волинська",
    "PXM": "Партнер Хмельницька",
    "PXN": "Партнер Херсонська",
    "PXV": "Партнер Харківська",
    "PZK": "Партнер Закарпатська",
    "PZP": "Партнер Запорізька",
}

# Reverse mapping: Ukrainian name to code
NAME_TO_CODE = {v: k for k, v in REGION_CODES.items()}

# City synonyms for search
CITY_SYNONYMS = {
    "київ": ["KI", "PKI"],
    "киев": ["KI", "PKI"],
    "kyiv": ["KI", "PKI"],
    "kiev": ["KI", "PKI"],
    "хмельницький": ["XM", "PXM"],
    "хмельницкий": ["XM", "PXM"],
    "khmelnytskyi": ["XM", "PXM"],
    "львів": ["LV", "PLV"],
    "львов": ["LV", "PLV"],
    "lviv": ["LV", "PLV"],
    "одеса": ["OD", "POD"],
    "одесса": ["OD", "POD"],
    "odesa": ["OD", "POD"],
    "харків": ["XV", "PXV"],
    "харьков": ["XV", "PXV"],
    "kharkiv": ["XV", "PXV"],
    "дніпро": ["DP", "PDP"],
    "днепр": ["DP", "PDP"],
    "dnipro": ["DP", "PDP"],
    "вінниця": ["VI", "PVI"],
    "винница": ["VI", "PVI"],
    "vinnytsia": ["VI", "PVI"],
    "запоріжжя": ["ZP", "PZP"],
    "запорожье": ["ZP", "PZP"],
    "zaporizhzhia": ["ZP", "PZP"],
    "полтава": ["PA", "PPA"],
    "poltava": ["PA", "PPA"],
    "чернігів": ["CN", "PCN"],
    "чернигов": ["CN", "PCN"],
    "chernihiv": ["CN", "PCN"],
    "черкаси": ["CK", "PCK"],
    "черкассы": ["CK", "PCK"],
    "cherkasy": ["CK", "PCK"],
    "тернопіль": ["TE", "PTE"],
    "тернополь": ["TE", "PTE"],
    "ternopil": ["TE", "PTE"],
    "рівне": ["RI", "PRI"],
    "ровно": ["RI", "PRI"],
    "rivne": ["RI", "PRI"],
    "миколаїв": ["MI", "PMI"],
    "николаев": ["MI", "PMI"],
    "mykolaiv": ["MI", "PMI"],
    "житомир": ["GT", "PGT"],
    "zhytomyr": ["GT", "PGT"],
    "суми": ["SM", "PSM"],
    "сумы": ["SM", "PSM"],
    "sumy": ["SM", "PSM"],
    "івано-франківськ": ["IF", "PIF"],
    "ивано-франковск": ["IF", "PIF"],
    "ivano-frankivsk": ["IF", "PIF"],
    "луцьк": ["VL", "PVL"],
    "луцк": ["VL", "PVL"],
    "lutsk": ["VL", "PVL"],
    "ужгород": ["ZK", "PZK"],
    "uzhhorod": ["ZK", "PZK"],
    "херсон": ["XN", "PXN"],
    "kherson": ["XN", "PXN"],
    "чернівці": ["CE", "PCE"],
    "черновцы": ["CE", "PCE"],
    "chernivtsi": ["CE", "PCE"],
    "кропивницький": ["KR", "PKR"],
    "кропивницкий": ["KR", "PKR"],
    "kropyvnytskyi": ["KR", "PKR"],
}


def get_region_name(code: str) -> str:
    """Get full Ukrainian name for region code."""
    return REGION_CODES.get(code, code)


def find_region_codes(query: str) -> list:
    """Find region codes matching a search query."""
    query_lower = query.lower()

    # Check direct synonyms
    for city, codes in CITY_SYNONYMS.items():
        if city in query_lower:
            return codes

    # Check if query contains region code
    for code in REGION_CODES.keys():
        if code.lower() in query_lower:
            return [code]

    # Check if query contains region name
    for code, name in REGION_CODES.items():
        if name.lower() in query_lower:
            return [code]

    return []


if __name__ == "__main__":
    # Test
    print("Region mapping test:")
    print(f"XM = {get_region_name('XM')}")
    print(f"KI = {get_region_name('KI')}")
    print()
    print("Search test:")
    print(f"'хмельницький' -> {find_region_codes('хмельницький')}")
    print(f"'клієнт з Києва' -> {find_region_codes('клієнт з Києва')}")
    print(f"'Львів' -> {find_region_codes('Львів')}")
