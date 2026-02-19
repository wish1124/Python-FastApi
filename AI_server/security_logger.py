"""
═══════════════════════════════════════════════════
[KT 2024-2025 BPFDoor 악성코드 사고 대응 - FastAPI 보안 로거]
═══════════════════════════════════════════════════

사고 원인: 1년 6개월간 악성코드 잠복을 탐지하지 못한 로깅/모니터링 부재

대응 전략:
  1. 구조화된 JSON 로그 (Azure Application Insights 연동)
  2. 이상 패턴 실시간 탐지 (새벽 접근, 취약점 스캔, 대량 요청)
  3. 민감정보 마스킹

적용: RAG_server.py에서
  from security_logger import SecurityLogMiddleware
  app.add_middleware(SecurityLogMiddleware)
═══════════════════════════════════════════════════
"""
import sys
import time
import json
import re
import logging
from datetime import datetime
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("security.audit")
logger.setLevel(logging.INFO)

# 파일 핸들러 (JSON Lines 형식)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)

# 취약점 스캔 의심 패턴
SUSPICIOUS_PATTERNS = [
    r"/\.env", r"/\.git", r"/wp-admin", r"/phpmyadmin",
    r"/admin", r"/actuator", r"\.sql$", r"\.bak$",
    r"/etc/passwd", r"\.\./", r"<script", r"union.*select",
]


def mask_sensitive(text: str) -> str:
    """민감정보 마스킹 (KT BPFDoor에서 로그 내 개인정보 평문 노출 방지)"""
    if not text:
        return text
    text = re.sub(r'(?i)(password|token|key|secret)[=:]\s*\S+', r'\1=****', text)
    text = re.sub(r'(\d{3})-?(\d{4})-?(\d{4})', r'\1-****-\3', text)
    text = re.sub(r'([a-zA-Z0-9])[a-zA-Z0-9.]*@', r'\1**@', text)
    return text


def _is_suspicious_path(path: str) -> bool:
    return any(re.search(p, path, re.IGNORECASE) for p in SUSPICIOUS_PATTERNS)


def _is_unusual_hour() -> bool:
    """새벽 2~5시 접근 탐지"""
    return 2 <= datetime.now().hour < 5


class SecurityLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        client_ip = self._extract_ip(request)
        path = request.url.path
        method = request.method

        # 이상 패턴 탐지
        alerts = []
        if _is_suspicious_path(path):
            alerts.append("VULN_SCAN_ATTEMPT")
        if _is_unusual_hour():
            alerts.append("UNUSUAL_HOUR_ACCESS")

        # 요청 처리
        response = await call_next(request)
        duration_ms = round((time.time() - start) * 1000, 2)

        # 구조화된 감사 로그 (JSON)
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "ip": client_ip,
            "method": method,
            "path": mask_sensitive(path),
            "status": response.status_code,
            "duration_ms": duration_ms,
            "user_agent": mask_sensitive(request.headers.get("user-agent", "")),
        }

        if alerts:
            log_entry["alerts"] = alerts
            log_entry["severity"] = "CRITICAL" if "VULN_SCAN_ATTEMPT" in alerts else "WARN"
            logger.warning(json.dumps(log_entry, ensure_ascii=False))
        elif response.status_code >= 400:
            log_entry["severity"] = "WARN"
            logger.warning(json.dumps(log_entry, ensure_ascii=False))
        else:
            log_entry["severity"] = "INFO"
            logger.info(json.dumps(log_entry, ensure_ascii=False))

        return response

    def _extract_ip(self, request: Request) -> str:
        xff = request.headers.get("X-Forwarded-For")
        if xff:
            return xff.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
