import asyncio
import contextvars
import json
import os
import random
import re
import uuid
from typing import Any

from simple_agentic_search.tools.shared import (
    DeepResearchTool,
    get_cache_async,
    get_cache_key,
    log_tool_event,
    run_with_retries_async,
    set_cache_async,
    shorten_for_log,
    track_api_usage,
)


_CTX_VISIT_MATCH_LOG: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "visit_match_log", default=None
)
_CTX_MODEL_REQUEST_PREFIX: contextvars.ContextVar[str] = contextvars.ContextVar(
    "visit_model_request_prefix", default="visit"
)


class VisitTool(DeepResearchTool):
    """Web page visiting with content extraction."""

    MAX_URLS = 3
    MAX_CONTENT_CHARS = 120000

    def _get_match_log(self) -> list:
        log = _CTX_VISIT_MATCH_LOG.get()
        if log is None:
            log = []
            _CTX_VISIT_MATCH_LOG.set(log)
        return log

    def get_match_log(self) -> list[dict]:
        return list(self._get_match_log())

    def reset_match_log(self) -> None:
        _CTX_VISIT_MATCH_LOG.set([])

    EXTRACTOR_PROMPT = """Summarize the following webpage content with respect to the user's goal.

Webpage Content (may be truncated):
{webpage_content}

User Goal: {goal}

Instructions:
- Summarize the main content of the webpage in no more than five sentences.
- Your summary should cover the overall key points of the page, not just parts related to the user's goal.
- If any part of the content is helpful for answering the user's goal, be sure to include it clearly in the summary. Do not ignore relevant information.
- If the page contains no information relevant to the goal, state that briefly.
- Your summary should be concise, factual, and informative.
- Return the summary text directly. Do not use JSON, code fences, or any special formatting.
"""

    def __init__(self, tools_config: dict | None = None):
        cfg = tools_config or {}
        self.MAX_URLS = int(cfg.get("visit_max_urls", self.MAX_URLS))
        self.MAX_CONTENT_CHARS = int(cfg.get("visit_max_content_chars", self.MAX_CONTENT_CHARS))
        super().__init__(
            name="visit",
            description="Visit webpage(s) and return the summary of the content.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": ["string", "array"],
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": self.MAX_URLS,
                        "description": (
                            "The URL(s) of the webpage(s) to visit. Can be a single URL "
                            "or an array of URLs."
                        ),
                    },
                    "goal": {
                        "type": "string",
                        "description": "The goal of the visit for webpage(s).",
                    },
                },
                "required": ["url", "goal"],
            },
        )
        self.api_max_retries = int(cfg.get("api_max_retries", 3))
        self.jina_api_key = os.getenv("JINA_API_KEY", "")
        self.jina_reader_url = os.getenv("READER_URL", "https://r.jina.ai")
        self.extract_model = os.getenv("EXTRACT_MODEL") or "self"
        self.extract_max_tokens = int(cfg.get("extract_max_tokens", os.getenv("EXTRACT_MAX_TOKEN", 8192)))
        raw_extract_urls = os.getenv("EXTRACT_URL", "")
        self.extract_urls = [
            item.strip() for item in raw_extract_urls.split(",") if item.strip()
        ]
        self.extract_key = os.getenv("EXTRACT_KEY", "")
        self._model_caller = None  # Set via set_model_caller() for self-extract mode
        self._model_timeout = None
        self._model_request_prefix = "visit"

    def set_model_caller(self, caller):
        """Set the rollout engine for self-summarization (when EXTRACT_MODEL=self)."""
        self._model_caller = caller

    def set_model_timeout(self, timeout):
        """Disable self-extract model timeouts; visit summaries should finish."""
        self._model_timeout = None

    def set_model_request_prefix(self, prefix: str):
        """Set per-sample request prefix for self-extract model calls."""
        safe_prefix = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(prefix or "visit"))[:160]
        self._model_request_prefix = safe_prefix or "visit"
        _CTX_MODEL_REQUEST_PREFIX.set(self._model_request_prefix)

    def _next_self_extract_request_id(self) -> str:
        prefix = _CTX_MODEL_REQUEST_PREFIX.get() or self._model_request_prefix
        return f"{prefix}:visit-summary:{uuid.uuid4().hex}"

    async def _abort_self_extract_request(self, request_id: str, reason: str) -> None:
        abort = getattr(self._model_caller, "abort_request", None)
        if abort is None:
            log_tool_event(
                "Visit/SelfExtract",
                "AbortUnavailable",
                f"request_id={request_id} reason={reason}",
                level="WARNING",
            )
            return
        try:
            await abort(request_id)
            log_tool_event(
                "Visit/SelfExtract",
                "Aborted",
                f"request_id={request_id} reason={reason}",
                level="WARNING",
            )
        except Exception as exc:  # noqa: BLE001
            log_tool_event(
                "Visit/SelfExtract",
                "AbortFailed",
                f"request_id={request_id} reason={reason} error={exc}",
                level="WARNING",
            )

    @property
    def _is_self_extract_mode(self) -> bool:
        """Check if extract mode uses the training model itself."""
        return self.extract_model.strip().lower() == "self"

    async def call(self, url: str | list, goal: str = "", **kwargs) -> str:
        """Visit webpages via Reader API and optionally summarize with a local model."""

        urls = [url] if isinstance(url, str) else list(url)
        if not urls:
            return "[Visit] No valid URL provided"

        warnings = []
        canonical_urls = []
        seen = set()
        duplicate_count = 0
        for raw_url in urls:
            normalized = self._normalize_url(str(raw_url))
            canonical = self._canonicalize_url(normalized)
            if canonical in seen:
                duplicate_count += 1
                continue
            seen.add(canonical)
            canonical_urls.append(canonical)

        if duplicate_count:
            warnings.append(
                f"[Visit Warning] Removed {duplicate_count} duplicate URL(s) from this visit call."
            )
        if len(canonical_urls) > self.MAX_URLS:
            warnings.append(
                f"[Visit Warning] Received {len(canonical_urls)} unique URLs; "
                f"only first {self.MAX_URLS} will be executed."
            )
        canonical_urls = canonical_urls[: self.MAX_URLS]

        tasks = [
            self._handle_single_url(target_url, goal)
            for target_url in canonical_urls
        ]
        results = await asyncio.gather(*tasks) if tasks else []

        return "\n\n=======\n\n".join(warnings + results)

    async def _handle_single_url(self, url: str, goal: str) -> str:
        normalized_url = self._normalize_url(url)
        # [DEBUG] Set env VISIT_DISABLED=1 to skip visit content for ablation
        # return f"The useful information in {normalized_url} for user goal {goal or 'N/A'} as follows:\n\nNo content retrieved (visit disabled for testing).\n"

        try:
            reader_payload = await self._fetch_reader_content(normalized_url)
        except Exception as exc:  # noqa: BLE001
            log_tool_event(
                source="Visit/Reader",
                status="Exception",
                message=f"url={normalized_url}",
                error=str(exc),
                level="ERROR",
            )
            return self._build_failure_message(
                normalized_url, goal, f"Unable to fetch webpage content: {exc}"
            )

        if reader_payload is None:
            return self._build_failure_message(
                normalized_url, goal, "Reader API returned empty payload"
            )

        content = reader_payload.get("content") or ""
        description = reader_payload.get("description") or ""

        if not content:
            fallback = description or "Webpage content is empty"
            return self._build_failure_message(normalized_url, goal, fallback)

        content = self._truncate_content(content)

        summary_result = await self._summarize_with_extract(
            content, goal, reader_payload
        )

        if summary_result is None:
            log_tool_event(
                "Visit", "ExtractSummaryFailed", f"url={normalized_url}", level="ERROR"
            )
            # Fall back to partial raw content with guidance
            raw_snippet = content[:5000] if content else (description or "Summary service unavailable.")
            summary_text = (
                f"Summarization failed. Below is a partial excerpt of the raw webpage content "
                f"(first {len(raw_snippet)} chars) for your reference:\n\n{raw_snippet}"
            )
        else:
            summary_text = summary_result

        return self._format_success(normalized_url, goal, summary_text)

    def _normalize_url(self, url: str) -> str:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if not parsed.scheme:
            return f"https://{url}"
        return url

    def _canonicalize_url(self, url: str) -> str:
        from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

        parsed = urlparse(url)
        scheme = parsed.scheme.lower() or "https"
        netloc = parsed.netloc.lower()
        path = parsed.path or "/"
        if path != "/":
            path = path.rstrip("/")
        query_items = parse_qsl(parsed.query, keep_blank_values=True)
        query = urlencode(query_items, doseq=True)
        return urlunparse((scheme, netloc, path, "", query, ""))

    def _select_extract_url(self) -> str | None:
        if not self.extract_urls:
            return None
        selected = random.choice(self.extract_urls)
        if not re.search(r"/v1/chat/completions/?$", selected):
            selected = f"{selected.rstrip('/')}/v1/chat/completions"
        return selected

    async def _fetch_reader_content(self, url: str) -> dict[str, Any] | None:
        # Check cache first
        cache_key = get_cache_key(url)
        cached_result = await get_cache_async(
            "text_visit", cache_key, executor=self.executor
        )
        if cached_result:
            try:
                result = json.loads(cached_result)
                self._get_match_log().append({"url": url, "source": "cache"})
                return result
            except json.JSONDecodeError:
                pass  # Continue with API call if cache is corrupted

        try:
            import requests
        except ImportError as exc:  # noqa: PERF203
            raise RuntimeError("Visit tool requires 'requests' package") from exc

        proxies = self._get_requests_proxies()
        headers = {}
        if self.jina_api_key:
            headers["Authorization"] = self.jina_api_key

        body = {"url": url}

        def send_request():
            return requests.post(
                self.jina_reader_url,
                headers=headers,
                data=body,
                timeout=60,
                proxies=proxies,
            )

        response = await run_with_retries_async(
            send_request, executor=self.executor,
            attempts=self.api_max_retries,
            source="VisitTool/JinaReader",
        )

        if response.status_code != 200:
            raise RuntimeError(f"Reader API returned HTTP {response.status_code}")

        track_api_usage("visit")
        self._get_match_log().append({"url": url, "source": "online"})

        result = {
            "content": response.text or "",
            "description": "",
            "meta": {
                "provider": "jina",
                "url": url,
                "reader_url": self.jina_reader_url,
            },
        }

        # Store result in cache only if we have valid content
        if result["content"].strip():
            await set_cache_async(
                "text_visit",
                cache_key,
                url,
                json.dumps(result, ensure_ascii=False),
                executor=self.executor,
            )

        return result

    def _truncate_content(self, content: str) -> str:
        if len(content) <= self.MAX_CONTENT_CHARS:
            return content
        return content[: self.MAX_CONTENT_CHARS] + "\n[Content truncated...]"

    async def _summarize_with_extract(
        self, content: str, goal: str, reader_payload: dict[str, Any]
    ) -> str | None:
        """Route to the appropriate summarization backend based on EXTRACT_MODEL."""
        if self._is_self_extract_mode:
            return await self._summarize_with_self(content, goal)
        return await self._summarize_with_external_api(content, goal, reader_payload)

    async def _summarize_with_self(self, content: str, goal: str) -> str | None:
        """Use the agentic model (rollout_engine) for summarization.

        Safe for concurrent calls: get_model_response operates on local variables
        and the inference server handles concurrent requests natively.
        """
        if self._model_caller is None:
            log_tool_event(
                "Visit/SelfExtract", "NoModelCaller",
                "EXTRACT_MODEL=self but no model_caller set, falling back to raw content",
                level="WARNING",
            )
            return None

        prompt = self.EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal or "N/A")
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Your task is to summarize webpage content concisely and factually."},
            {"role": "user", "content": prompt},
        ]
        request_id = self._next_self_extract_request_id()
        try:
            coro = self._model_caller.get_model_response(
                messages=messages,
                max_tokens=self.extract_max_tokens,
                temperature=0.0,
                top_p=1.0,
                presence_penalty=0.1,
                application_id=request_id,
            )
            response = (
                await asyncio.wait_for(coro, timeout=self._model_timeout)
                if self._model_timeout
                else await coro
            )
            text = (response.content or response.text or "").strip()
            if text:
                return text
            log_tool_event("Visit/SelfExtract", "EmptyResponse", "", level="WARNING")
            return None
        except asyncio.TimeoutError:
            await self._abort_self_extract_request(request_id, "timeout")
            log_tool_event(
                "Visit/SelfExtract",
                "Timeout",
                f"request_id={request_id} timeout={self._model_timeout}",
                level="WARNING",
            )
            return None
        except asyncio.CancelledError:
            await self._abort_self_extract_request(request_id, "cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            log_tool_event("Visit/SelfExtract", "Failed", f"error={exc}", level="WARNING")
            return None

    async def _summarize_with_external_api(
        self, content: str, goal: str, reader_payload: dict[str, Any]
    ) -> str | None:
        """Call external extract API to summarize webpage content. Returns plain text summary or None."""
        extract_url = self._select_extract_url()
        if not extract_url:
            log_tool_event(
                source="Visit/Extract",
                status="Config",
                message="EXTRACT_URL is not set, skip extract service",
            )
            return None

        try:
            import requests
        except ImportError:
            log_tool_event(
                source="Visit/Extract",
                status="DependencyMissing",
                message="'requests' package not installed, cannot call extract service",
                level="WARNING",
            )
            return None

        prompt = self.EXTRACTOR_PROMPT.format(
            webpage_content=content, goal=goal or "N/A"
        )

        extract_messages = [
            {"role": "system", "content": "You are a helpful assistant. Your task is to summarize webpage content concisely and factually."},
            {"role": "user", "content": prompt},
        ]

        payload = {
            "model": self.extract_model or "default",
            "messages": extract_messages,
            "max_tokens": self.extract_max_tokens,
        }

        headers = {"Content-Type": "application/json"}
        if self.extract_key:
            headers["Authorization"] = self.extract_key
        proxies = self._get_requests_proxies()

        try:
            response = await run_with_retries_async(
                lambda: requests.post(
                    url=extract_url,
                    headers=headers,
                    json=payload,
                    timeout=60,
                    proxies=proxies,
                ),
                executor=self.executor,
                attempts=self.api_max_retries,
                source="VisitTool/Extract",
            )
        except Exception as exc:  # noqa: BLE001
            log_tool_event(
                source="Visit/Extract",
                status="RequestError",
                message=f"url={extract_url}",
                error=str(exc),
                level="ERROR",
            )
            return None

        if response.status_code != 200:
            log_tool_event(
                source="Visit/Extract",
                status="HTTPError",
                message=f"url={extract_url} status={response.status_code}",
                level="WARNING",
            )
            return None

        try:
            result = response.json()
        except json.JSONDecodeError:
            log_tool_event(
                source="Visit/Extract",
                status="ParseError",
                message="Extract service returned non-JSON response, unable to parse",
                level="WARNING",
            )
            return None

        # Track extract model usage
        if isinstance(result, dict):
            usage = result.get("usage", {})
            if isinstance(usage, dict):
                total_tokens = usage.get("total_tokens", 0) or (
                    usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                )
                track_api_usage("extract", tokens=total_tokens)
            else:
                track_api_usage("extract")

        # Extract text content from OpenAI-compatible response
        summary_text: str | None = None

        if isinstance(result, dict):
            choices = result.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0] or {}
                if isinstance(first_choice, dict):
                    message_dict = first_choice.get("message")
                    if isinstance(message_dict, dict):
                        message_content = message_dict.get("content")
                        if isinstance(message_content, str) and message_content.strip():
                            summary_text = message_content.strip()
                    if summary_text is None:
                        text_candidate = first_choice.get("text")
                        if isinstance(text_candidate, str) and text_candidate.strip():
                            summary_text = text_candidate.strip()
            if summary_text is None:
                fallback_payload = result.get("content") or result.get("data")
                if isinstance(fallback_payload, str) and fallback_payload.strip():
                    summary_text = fallback_payload.strip()

        if not summary_text:
            log_tool_event(
                source="Visit/Extract",
                status="InvalidContent",
                message="Extract service response missing usable content",
                level="WARNING",
            )
            return None

        return summary_text

    def _build_failure_message(self, url: str, goal: str, reason: str) -> str:
        useful_information = (
            f"The useful information in {url} for user goal {goal or 'N/A'} as follows:\n\n"
            f"Unable to retrieve webpage content: {reason}\n"
        )

        reason_preview = shorten_for_log(reason)
        result_preview = shorten_for_log(useful_information)
        log_tool_event(
            source="Visit",
            status="Failure",
            message=(
                f"url={url} "
                f"reason_len={len(reason)} "
                f"result_len={len(useful_information)} "
                f"reason_preview={json.dumps(reason_preview, ensure_ascii=False)} "
                f"result_preview={json.dumps(result_preview, ensure_ascii=False)}"
            ),
            level="WARNING",
        )

        return useful_information

    def _format_success(self, url: str, goal: str, summary: str) -> str:
        useful_information = (
            f"The useful information in {url} for user goal {goal or 'N/A'} as follows:\n\n"
            f"{summary or 'No summary generated'}\n"
        )

        summary_preview = shorten_for_log(summary or "")
        return useful_information


if __name__ == "__main__":
    import argparse
    import os
    import time
    from pathlib import Path

    # Load .env file
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ.setdefault(key, value)
        print(f"Loaded env from {env_path}")

    parser = argparse.ArgumentParser(description="Test VisitTool timing")
    parser.add_argument(
        "--url", type=str, nargs="+", default="https://en.wikipedia.org/wiki/Eiffel_Tower",
        help="URL(s) to visit, e.g. --url 'https://en.wikipedia.org/wiki/Eiffel_Tower'",
    )
    parser.add_argument("--goal", type=str, default="Extract key information from the page")
    args = parser.parse_args()

    async def _test():
        from concurrent.futures import ThreadPoolExecutor
        from simple_agentic_search.tools.shared import ensure_cache_initialized

        await ensure_cache_initialized()

        tool = VisitTool()
        executor = ThreadPoolExecutor(max_workers=8)
        tool.set_executor(executor)

        print(f"=== VisitTool Test ===")
        print(f"URLs: {args.url}")
        print(f"Goal: {args.goal}")
        print()

        # Test each URL with step-level timing
        urls = args.url if isinstance(args.url, list) else [args.url]
        for i, url in enumerate(urls):
            normalized = tool._normalize_url(url)

            # Step 1: Reader API
            t0 = time.time()
            try:
                reader_payload = await tool._fetch_reader_content(normalized)
            except Exception as e:
                print(f"[URL {i+1}] Reader failed: {e}")
                continue
            t1 = time.time()
            content_len = len(reader_payload.get("content", "")) if reader_payload else 0
            print(f"[URL {i+1}] Reader fetch:  {t1 - t0:.2f}s  -> content {content_len} chars")

            if not reader_payload or not reader_payload.get("content"):
                print(f"[URL {i+1}] No content, skipping extract.")
                continue

            content = tool._truncate_content(reader_payload["content"])

            # Step 2: Extract summarization
            t2 = time.time()
            summary = await tool._summarize_with_extract(content, args.goal, reader_payload)
            t3 = time.time()
            if summary:
                s_preview = summary.replace("\n", " ")
                print(f"[URL {i+1}] Extract:       {t3 - t2:.2f}s  -> {s_preview}...")
            else:
                print(f"[URL {i+1}] Extract:       {t3 - t2:.2f}s  -> FAILED")
            print()

        # Full end-to-end
        t_start = time.time()
        full_result = await tool.call(url=args.url, goal=args.goal)
        t_end = time.time()
        print(f"[Full E2E] Total:       {t_end - t_start:.2f}s")
        print(f"[Full E2E] Result length: {len(full_result)} chars")
        print(full_result)

        executor.shutdown(wait=False)

    asyncio.run(_test())
