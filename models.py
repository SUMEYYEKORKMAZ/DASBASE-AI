"""DASBase arşiv parçası (chunk) veri modelleri."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    document_type: str = ""
    department: str = ""
    security_level: str = ""
    sdp: str = ""
    file_no: str = ""
    allowed_roles: list[str] = Field(default_factory=list)
    page: int = 1
    text: str = ""
    score: float = 0.95
    similarity: float = 0.95
    distance: float = 0.05
    source_rank: str = ""

    model_config = {"extra": "allow"}

    def roles_csv(self) -> str:
        return ",".join(self.allowed_roles)

    def display_similarity(self) -> float:
        """Kart yüzdesi için 0-1 arası skor; yoksa 0.95."""
        for raw in (self.similarity, self.score):
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if 0.0 < value <= 1.0:
                return value
        try:
            distance = float(self.distance)
            mapped = 1.0 - distance
            if 0.0 < mapped <= 1.0:
                return mapped
        except (TypeError, ValueError):
            pass
        return 0.95


class QueryResult(BaseModel):
    answer: str
    hits: list[SearchHit] = Field(default_factory=list)
    search_type: str
    role_code: str
    pii_redactions: int = 0
    used_mock_llm: bool = False
    evaluation: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)
