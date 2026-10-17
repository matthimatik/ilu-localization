import re

ISO_6346_ID_REGEX = re.compile(r"^[A-Z]{4}\d{7}$")

OWNER_CHARACTER_TO_NUMBER = {
    "A": 10,
    "B": 12,
    "C": 13,
    "D": 14,
    "E": 15,
    "F": 16,
    "G": 17,
    "H": 18,
    "I": 19,
    "J": 20,
    "K": 21,
    "L": 23,
    "M": 24,
    "N": 25,
    "O": 26,
    "P": 27,
    "Q": 28,
    "R": 29,
    "S": 30,
    "T": 31,
    "U": 32,
    "V": 34,
    "W": 35,
    "X": 36,
    "Y": 37,
    "Z": 38,
}


def normalize(container_code: str) -> str:
    return container_code.replace(" ", "").upper()


def calculate_check_digit(owner_and_serial: str) -> int:
    total = 0
    for i, char in enumerate(owner_and_serial.upper()):
        if char.isdigit():
            weight = int(char)
        else:
            weight = OWNER_CHARACTER_TO_NUMBER[char]
        total += weight * (2**i)
    remainder = total % 11
    return 0 if remainder == 10 else remainder


def validate_iso6346(container_number: str) -> tuple[bool, str]:
    code = normalize(container_number)
    if len(code) != 11:
        return False, "Invalid length"
    if not ISO_6346_ID_REGEX.match(code):
        return False, "Invalid pattern"
    computed = calculate_check_digit(code[:10])
    if computed != int(code[10]):
        return False, f"Invalid check digit (expected {computed})"
    return True, "Valid"
