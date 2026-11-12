import asyncio
import contextvars
import json
import os

from simple_agentic_search.tools.shared import (
    DeepResearchTool,
    get_cache_async,
    get_cache_key,
    log_search,
    log_tool_event,
    run_with_retries_async,
    set_cache_async,
    track_api_usage,
)


_CTX_SEARCH_MATCH_LOG: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "text_search_match_log", default=None
)


class SearchTool(DeepResearchTool):
    """Web search tool using Serper API."""

    MAX_URLS = 3
    MAX_QUERIES = 3
    CACHE_MAX_URLS = 20

    def _get_match_log(self) -> list:
        log = _CTX_SEARCH_MATCH_LOG.get()
        if log is None:
            log = []
            _CTX_SEARCH_MATCH_LOG.set(log)
        return log

    def get_match_log(self) -> list[dict]:
        return list(self._get_match_log())

    def reset_match_log(self) -> None:
        _CTX_SEARCH_MATCH_LOG.set([])

    def __init__(self, tools_config: dict | None = None):
        cfg = tools_config or {}
        self.MAX_URLS = int(cfg.get("search_max_urls", self.MAX_URLS))
        self.MAX_QUERIES = int(cfg.get("search_max_queries", self.MAX_QUERIES))
        self.CACHE_MAX_URLS = int(cfg.get("search_cache_max_urls", self.CACHE_MAX_URLS))
        super().__init__(
            name="text_search",
            description=(
                "Performs batched web searches: supply complementary queries in a single call; "
                "the tool retrieves the top results for each query. "
                "Call as {\"name\": \"text_search\", \"arguments\": {\"query\": [\"...\"]}}; "
                "the query array must stay inside arguments."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": self.MAX_QUERIES,
                        "description": (
                            "Array of query strings. "
                            "Include multiple complementary search queries in a single call when useful."
                        ),
                    },
                },
                "required": ["query"],
            },
        )
        self.api_max_retries = int(cfg.get("api_max_retries", 3))
        self.serp_api_key = os.getenv("SERP_API_KEY")
        self.serp_search_url = os.getenv(
            "TEXT_SEARCH_URL", "https://google.serper.dev/search"
        )

    async def _serp_search(self, query: str | list) -> str:
        """Use Serp web search API when key is available."""
        try:
            import requests
        except ImportError:
            return """[Search - Dependencies Required]

Please install requests: pip install requests"""

        queries = [query] if isinstance(query, str) else query
        proxies = self._get_requests_proxies()

        def normalize_items(data_obj) -> list[dict]:
            items = []
            if isinstance(data_obj, dict):
                items = data_obj.get("organic") or []
            normalized = []
            for item in items[: self.CACHE_MAX_URLS]:
                if not isinstance(item, dict):
                    continue
                normalized.append(
                    {
                        "title": item.get("title", "Untitled"),
                        "url": item.get("link", ""),
                        "snippet": (item.get("snippet", "") or "").strip(),
                        "date": item.get("date"),
                    }
                )
            return normalized

        def format_items(q: str, items: list[dict]) -> str:
            display_items = items[: self.MAX_URLS]
            web_snippets = []
            for idx, item in enumerate(display_items, 1):
                title = item.get("title", "Untitled") or "Untitled"
                url = item.get("url", "") or ""
                snippet = (item.get("snippet", "") or "").strip()
                date = item.get("date")

                entry = f"{idx}. [{title}]({url})"
                if date:
                    entry += f"\n   Date published: {date}"
                if snippet:
                    entry += f"\n   {snippet}"
                web_snippets.append(entry)

            return (
                f"Search for '{q}' returned {len(web_snippets)} results:\n\n"
                + "\n\n".join(web_snippets)
                if web_snippets
                else f"No search results found for '{q}'"
            )

        def encode_cache(items: list[dict]) -> str:
            return json.dumps({"version": 1, "items": items}, ensure_ascii=False)

        def decode_cache(cached_result):
            if not isinstance(cached_result, str):
                return None
            try:
                payload = json.loads(cached_result)
            except json.JSONDecodeError:
                return None
            if isinstance(payload, dict) and isinstance(payload.get("items"), list):
                return payload["items"]
            return None

        async def search_single_query(q: str) -> str:
            cache_key = get_cache_key(q)
            cached_result = await get_cache_async(
                "text_search", cache_key, executor=self.executor
            )
            if cached_result:
                self._get_match_log().append({"query": q, "source": "cache"})
                cached_items = decode_cache(cached_result)
                if cached_items is not None:
                    return format_items(q, cached_items)
                return cached_result

            payload = {
                "q": q,
                "hl": "en",
                "gl": "us",
            }

            headers = {
                "X-API-KEY": self.serp_api_key,
                "Content-Type": "application/json",
            }

            def send_request():
                return requests.post(
                    self.serp_search_url,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False),
                    timeout=300,
                    proxies=proxies,
                )

            try:
                resp = await run_with_retries_async(
                    send_request, executor=self.executor,
                    attempts=self.api_max_retries,
                    source="SearchTool/Serp",
                )
            except Exception as exc:  # noqa: BLE001
                error_message = f"Search request failed for '{q}': {exc}"
                log_search("Serp", "Exception", q, error=error_message)
                return error_message

            # Track: 1 text search = 1 credit
            track_api_usage("text_search")
            self._get_match_log().append({"query": q, "source": "online"})

            text = resp.text
            try:
                data_obj = resp.json()
            except Exception:
                data_obj = None

            if resp.status_code != 200:
                error_message = f"HTTP {resp.status_code}: {text}"
                log_search("Serp", "HTTPError", q, error=error_message)
                return f"Search returned HTTP {resp.status_code} for '{q}'\n{text}"

            cache_items = normalize_items(data_obj)
            content = format_items(q, cache_items)
            # Cache a larger structured result set so search_max_urls can change
            # without being locked to the top-k from the first cache write.
            if cache_items:
                await set_cache_async(
                    "text_search", cache_key, q, encode_cache(cache_items), executor=self.executor
                )

            return content

        tasks = [search_single_query(q) for q in queries]
        all_results: list[str] = await asyncio.gather(*tasks) if tasks else []

        final_result = (
            "\n=======\n".join(all_results)
            if len(all_results) > 1
            else (all_results[0] if all_results else "")
        )

        return final_result

    async def call(self, query: str | list, **kwargs) -> str:
        """
        Search the web using Serper API.

        Args:
            query: Search query string or list of queries (max MAX_QUERIES)

        Returns:
            Formatted search results
        """
        # Normalize to list and enforce MAX_QUERIES with explicit feedback.
        if isinstance(query, str):
            query = [query]
        query = list(query)
        warnings = []
        if len(query) > self.MAX_QUERIES:
            warnings.append(
                f"[Search Warning] Received {len(query)} queries; "
                f"only first {self.MAX_QUERIES} will be executed."
            )
        query = query[: self.MAX_QUERIES]

        if not self.serp_api_key:
            message = f"""[Search - API Key Required]

To enable web search:
1. Get an API key from https://serper.dev
2. Add to .env: SERP_API_KEY=your_key_here

Placeholder results for '{query}'..."""

            log_tool_event("Search", "APIKeyMissing", f"query={query}", level="ERROR")
            log_search("Serp", "Config", str(query), error=message)
            if warnings:
                return "\n\n".join(warnings + [message])
            return message

        result = await self._serp_search(query)
        if warnings:
            return "\n\n".join(warnings + [result])
        return result


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

    parser = argparse.ArgumentParser(description="Test SearchTool timing")
    parser.add_argument(
        "--query", type=str, nargs="+", required=True,
        help="Search query string(s), e.g. --query 'Eiffel Tower' 'Statue of Liberty'",
    )
    args = parser.parse_args()

    async def _test():
        from concurrent.futures import ThreadPoolExecutor
        from simple_agentic_search.tools.shared import ensure_cache_initialized

        await ensure_cache_initialized()

        tool = SearchTool()
        executor = ThreadPoolExecutor(max_workers=8)
        tool.set_executor(executor)

        queries = args.query
        print(f"=== SearchTool Test ===")
        print(f"Queries: {queries}")
        print()

        # Test each query individually
        for i, q in enumerate(queries):
            t0 = time.time()
            result = await tool.call(query=[q])
            t1 = time.time()
            preview = result[:200].replace("\n", " ") if result else "EMPTY"
            print(f"[Query {i+1}] '{q}': {t1 - t0:.2f}s")
            print(f"         {preview}...")
            print()

        # Test batch
        if len(queries) > 1:
            t0 = time.time()
            result = await tool.call(query=queries)
            t1 = time.time()
            print(f"[Batch] All {len(queries)} queries: {t1 - t0:.2f}s, result length: {len(result)} chars")

        executor.shutdown(wait=False)

    asyncio.run(_test())
