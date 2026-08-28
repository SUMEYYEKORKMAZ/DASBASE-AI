"""KVKK / PII maskeleme: T.C. kimlik no, telefon ve şahıs isimleri."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Demo arşivindeki kişi adları (uzun adlar önce maskelenir).
DEMO_PERSON_NAMES = [
    "Ali Can Öztürk",
    "Ahmet Yılmaz",
    "Arzu Şahin",
    "Caner Yılmaz",
    "Mehmet Kaya",
    "Elif Aksoy",
    "Ayşe Demir",
    "Deniz Yücel",
    "Selin Arslan",
    "Burak Şahin",
    "Caner Polat",
    "Pınar Koç",
    "Hakan Erdem",
    "Cem Aydın",
]

_TC_RE = re.compile(r"(?<!\d)[1-9]\d{10}(?!\d)")
_PHONE_RE = re.compile(
    r"""
    (?:
        (?:\+90|0)\s*5\d{2}[\s./-]*\d{3}[\s./-]*\d{2}[\s./-]*\d{2}
        |
        0\s*[2-4]\d{2}[\s./-]*\d{3}[\s./-]*\d{2}[\s./-]*\d{2}
    )
    """,
    re.VERBOSE,
)
_TITLED_NAME_RE = re.compile(
    r"\b(?:Sn\.|Dr\.)\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+){1,2}"
)


@dataclass
class RedactionResult:
    text: str
    tc_count: int = 0
    phone_count: int = 0
    name_count: int = 0
    details: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.tc_count + self.phone_count + self.name_count


def redact_pii(text: str) -> RedactionResult:
    """LLM'e gitmeden önce kişisel verileri anonimleştirir."""
    if not text:
        return RedactionResult(text="")

    masked = text
    details: list[str] = []

    masked, tc_count = _TC_RE.subn("[MASKED_TC]", masked)
    if tc_count:
        details.append(f"T.C. Kimlik No x{tc_count}")

    masked, phone_count = _PHONE_RE.subn("[MASKED_PHONE]", masked)
    if phone_count:
        details.append(f"Telefon x{phone_count}")

    name_count = 0
    for person in sorted(DEMO_PERSON_NAMES, key=len, reverse=True):
        pattern = re.compile(re.escape(person), re.IGNORECASE)
        masked, found = pattern.subn("[MASKED_NAME]", masked)
        name_count += found

    titled, titled_count = _TITLED_NAME_RE.subn("[MASKED_NAME]", masked)
    masked = titled
    name_count += titled_count
    if name_count:
        details.append(f"Şahıs adı x{name_count}")

    return RedactionResult(
        text=masked,
        tc_count=tc_count,
        phone_count=phone_count,
        name_count=name_count,
        details=details,
    )


def has_unmasked_pii(text: str) -> bool:
    """Yanıtta maskelenmemiş T.C., telefon veya demo şahıs adı kalıp kalmadığını kontrol eder."""
    if not text:
        return False
    if _TC_RE.search(text) or _PHONE_RE.search(text):
        return True
    lowered = text.lower()
    return any(name.lower() in lowered for name in DEMO_PERSON_NAMES)


def redact_many(texts: list[str]) -> tuple[list[str], int]:
    results = [redact_pii(item) for item in texts]
    return [item.text for item in results], sum(item.total for item in results)
