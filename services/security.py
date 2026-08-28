"""Kullanıcı rolüne göre metadata filtreleme (RBAC mock)."""

from __future__ import annotations

from config import ROLE_EQUIVALENTS, ROLE_LABELS
from models import SearchHit


def resolve_role_code(role_label: str) -> str:
    """Arayüzdeki rol adını DASBase allowedRoles koduna çevirir."""
    return ROLE_LABELS.get(role_label, role_label)


def matching_role_codes(role_code: str) -> tuple[str, ...]:
    """Genel Müdür / Yonetici gibi eşdeğer kodları birlikte kabul eder."""
    return ROLE_EQUIVALENTS.get(role_code, (role_code,))


def parse_roles(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def role_flag_key(role_code: str) -> str:
    return f"role_{role_code}"


def chroma_where_filter(role_code: str) -> dict | None:
    """ChromaDB metadata filtresi. Genel Müdür tüm koleksiyonu görür."""
    if role_code in ("Genel_Mudur", "Yonetici"):
        return None
    codes = matching_role_codes(role_code)
    if len(codes) == 1:
        return {role_flag_key(codes[0]): True}
    return {"$or": [{role_flag_key(code): True} for code in codes]}


def is_allowed_for_role(allowed_roles: str | list[str], role_code: str) -> bool:
    if role_code in ("Genel_Mudur", "Yonetici"):
        return True
    allowed = set(parse_roles(allowed_roles))
    return any(code in allowed for code in matching_role_codes(role_code))


def filter_hits_by_role(hits: list[SearchHit], role_code: str) -> list[SearchHit]:
    """Chroma filtresine ek güvenlik ağı: Python tarafında RBAC."""
    return [hit for hit in hits if is_allowed_for_role(hit.allowed_roles, role_code)]
