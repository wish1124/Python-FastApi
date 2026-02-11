"""
═══════════════════════════════════════════════════
[KT 2014 홈페이지 해킹 사고 대응 - FastAPI Rate Limiting]
═══════════════════════════════════════════════════

사고 원인: API 요청 횟수 제한 없이 무차별 대입 공격 허용

대응 전략: IP 기반 Sliding Window Rate Limiting
  - /analyze: 5회/분 (AI 분석은 비용이 큰 연산)
  - /predictBase: 20회/분
  - 기본: 60회/분
  - 제한 초과 시 429 + Retry-After 헤더

적용: RAG_server.py에서
  from rate_limit import RateLimitMiddleware
  app.add_middleware(RateLimitMiddleware)
═══════════════════════════════════════════════════
"""

import time
import logging
from collections import defaultdict, deque
from typing import Dict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("security.rate_limit")

RATE_LIMITS: Dict[str, int] = {
    "/analyze": 5,
    "/predictBase": 20,
}
DEFAULT_LIMIT = 60
WINDOW_SECONDS = 60


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self._requests: Dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        client_ip = self._extract_ip(request)
        path = request.url.path
        limit = self._resolve_limit(path)
        key = f"{client_ip}:{self._categorize(path)}"
        now = time.time()
        window = self._requests[key]

        while window and window[0] < now - WINDOW_SECONDS:
            window.popleft()

        if len(window) >= limit:
            retry_after = int(window[0] + WINDOW_SECONDS - now) + 1
            if len(window) >= limit * 3:
                logger.warning(
                    "[SECURITY] 공격 의심 - IP: %s, Path: %s, 요청수: %d/분 (제한: %d)",
                    client_ip, path, len(window), limit
                )
            return JSONResponse(
                status_code=429,
                content={"status": "error", "message": f"요청 한도 초과. {retry_after}초 후 재시도해주세요."},
                headers={"Retry-After": str(retry_after)}
            )

        window.append(now)
        if len(self._requests) > 1000:
            self._cleanup(now)
        return await call_next(request)

    def _extract_ip(self, request: Request) -> str:
        xff = request.headers.get("X-Forwarded-For")
        if xff:
            return xff.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _resolve_limit(self, path: str) -> int:
        for prefix, limit in RATE_LIMITS.items():
            if path.startswith(prefix):
                return limit
        return DEFAULT_LIMIT

    def _categorize(self, path: str) -> str:
        for prefix in RATE_LIMITS:
            if path.startswith(prefix):
                return prefix
        return "DEFAULT"

    def _cleanup(self, now: float):
        expired = [k for k, v in self._requests.items() if not v or v[-1] < now - WINDOW_SECONDS]
        for k in expired:
            del self._requests[k]
