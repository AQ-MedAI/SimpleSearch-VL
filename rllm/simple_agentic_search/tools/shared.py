"""
DeepResearch Tools - Shared utilities
"""

import asyncio
import contextvars
import fcntl
import hashlib
import json
import logging
import os
import random
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, TypeVar, Union

# Async SQLite support (required dependency)
import aiosqlite

from rllm.tools.tool_base import Tool as RLLMTool

T = TypeVar("T")


@dataclass
class ToolCallResult:
    """Structured return value for tool calls that produce images alongside text.

    Using a return value (instead of side-effect instance attributes) keeps
    concurrent rollouts fully isolated — no shared mutable state between tasks.
    """
    text: str
    images: list = field(default_factory=list)
    image_urls: list = field(default_factory=list)


# ============================================================================
# Per-sample API usage tracker (contextvar-based, isolated per rollout)
# ============================================================================

_CTX_API_USAGE: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "api_usage", default=None
)


def reset_api_usage() -> None:
    """Reset API usage counters for a new sample/rollout."""
    _CTX_API_USAGE.set({
        "text_search": 0,
        "image_search": 0,
        "visit": 0,
        "extract": 0,
        "extract_tokens": 0,
    })


def track_api_usage(service: str, tokens: int = 0) -> None:
    """Increment API usage counter for a service.

    Args:
        service: "text_search" | "image_search" | "visit" | "extract"
        tokens: token count for extract model calls (prompt + completion).
    """
    usage = _CTX_API_USAGE.get()
    if usage is None:
        return
    if service in usage:
        usage[service] += 1
    if tokens and "extract_tokens" in usage:
        usage["extract_tokens"] += tokens


def get_api_usage() -> dict:
    """Get current API usage counters for this rollout."""
    return dict(_CTX_API_USAGE.get() or {})


def _normalize_level(level: str | None) -> str:
    if not level:
        return "INFO"
    return str(level).upper()


def run_with_retries(
    func: Callable[[], T],
    attempts: int = 5,
    delay: float = 0.5,
    source: str = "Unknown",
) -> T:
    """Execute a callable with retry support."""

    last_error: Exception | None = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= attempts:
                break
            # Log retry attempt
            log_tool_event(
                "Retry",
                "AttemptFailed",
                f"[{source}] Attempt {attempt}/{attempts} failed, will retry: {exc}",
                level="WARNING",
            )
            if delay > 0:
                time.sleep(delay)

    if last_error is not None:
        raise last_error

    raise RuntimeError("run_with_retries executed without performing any attempts")


async def run_blocking(
    func: Callable[[], T], executor: ThreadPoolExecutor | None = None
) -> T:
    """Run a blocking call in the given executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, func)


async def run_with_retries_async(
    func: Callable[[], T],
    attempts: int = 3,
    delay: float = 1,
    executor: ThreadPoolExecutor | None = None,
    source: str = "Unknown",
) -> T:
    """Execute a callable with retry support without blocking the event loop."""

    last_error: Exception | None = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            return await run_blocking(func, executor=executor)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= attempts:
                break
            # Log retry attempt
            log_tool_event(
                "Retry",
                "AttemptFailed",
                f"[{source}] Attempt {attempt}/{attempts} failed, will retry: {exc}",
                level="WARNING",
            )
            if delay > 0:
                await asyncio.sleep(delay)

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "run_with_retries_async executed without performing any attempts"
    )


def shorten_for_log(text: str, limit: int = 200) -> str:
    """Create a concise preview string for debug logging."""

    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    if not text:
        return ""

    normalized = text.replace("\n", "\\n")
    if len(normalized) <= limit * 2:
        return normalized
    return f"{normalized[:limit]} ... {normalized[-limit:]}"


def _select_extract_url(env_key: str = "EXTRACT_URL") -> str | None:
    raw_value = os.getenv(env_key, "")
    if not raw_value:
        return None
    candidates = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not candidates:
        return None
    selected = random.choice(candidates)
    if not re.search(r"/v1/chat/completions/?$", selected):
        selected = f"{selected.rstrip('/')}/v1/chat/completions"
    return selected


# Cache database configuration
CACHE_CONFIG = {
    "db_path": os.getenv("CACHE_DB_PATH", "simple_agentic_search/cache/mars_cache.db"),
    "max_age_days": int(os.getenv("CACHE_MAX_AGE_DAYS", "30")),  # Cache validity period
    "max_retries": int(os.getenv("CACHE_MAX_RETRIES", "3")),  # Max retry attempts
    "base_retry_delay": float(
        os.getenv("CACHE_RETRY_DELAY", "0.1")
    ),  # Base delay in seconds
    "busy_timeout": int(
        os.getenv("CACHE_BUSY_TIMEOUT", "1000")
    ),  # SQLite busy timeout in ms
    "process_lock_enabled": os.getenv("CACHE_PROCESS_LOCK", "1").lower()
    not in {"0", "false", "no"},
    "process_lock_mode": os.getenv("CACHE_PROCESS_LOCK_MODE", "shared").lower(),
    "process_lock_timeout": float(os.getenv("CACHE_PROCESS_LOCK_TIMEOUT", "0")),
}

# Global async database connection manager
_async_db_pool: Optional["AsyncCacheDB"] = None
_async_db_lock = asyncio.Lock()


class AsyncCacheDB:
    """
    Async SQLite database connection manager.

    Optimized for high-concurrency scenarios:
    - Use WAL mode to enable concurrent reads
    - Protect writes with a lock to avoid conflicts
    - Single-connection design to avoid SQLite lock contention
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None
        self._process_lock_file = None
        self._init_lock = asyncio.Lock()  # Initialization lock
        self._write_lock = asyncio.Lock()  # Write lock to protect concurrent writes
        self._initialized = False

    async def get_connection(self) -> aiosqlite.Connection:
        """Get a database connection (lazy initialization)."""
        if self._connection is None:
            async with self._init_lock:
                if self._connection is None:
                    await self._create_connection()
        return self._connection

    async def _create_connection(self):
        """Create and configure the database connection."""
        # Ensure the database directory exists.
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        await self._acquire_process_lock()

        try:
            self._connection = await aiosqlite.connect(self.db_path)
        except Exception:
            self._release_process_lock()
            raise

        # Configure database options for high concurrency.
        await self._connection.execute(
            "PRAGMA journal_mode=WAL"
        )  # WAL enables concurrent reads
        await self._connection.execute(
            "PRAGMA synchronous=NORMAL"
        )  # Balance performance and durability
        await self._connection.execute("PRAGMA cache_size=-64000")  # 64MB cache
        await self._connection.execute(
            f"PRAGMA busy_timeout={CACHE_CONFIG['busy_timeout']}"
        )
        await self._connection.execute("PRAGMA wal_autocheckpoint=1000")
        await self._connection.execute("PRAGMA temp_store=MEMORY")
        await self._connection.execute(
            "PRAGMA read_uncommitted=1"
        )  # Allow dirty reads to improve concurrency

        # Initialize tables.
        if not self._initialized:
            await self._initialize_tables()
            self._initialized = True

    async def _acquire_process_lock(self):
        """Acquire a cooperative process lock before opening the cache DB.

        Shared mode is Ray-friendly: many rollout workers can use the same DB.
        Exclusive mode is for maintenance jobs that must wait for active users.
        """
        if not CACHE_CONFIG["process_lock_enabled"]:
            return
        lock_mode = CACHE_CONFIG["process_lock_mode"]
        if lock_mode not in {"shared", "exclusive"}:
            raise ValueError(
                "CACHE_PROCESS_LOCK_MODE must be 'shared' or 'exclusive', "
                f"got {lock_mode!r}"
            )
        lock_path = f"{self.db_path}.lock"
        lock_file = open(lock_path, "a+", encoding="utf-8")
        lock_flag = fcntl.LOCK_EX if lock_mode == "exclusive" else fcntl.LOCK_SH
        timeout = CACHE_CONFIG["process_lock_timeout"]
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(lock_file.fileno(), lock_flag | fcntl.LOCK_NB)
                if lock_mode == "exclusive":
                    lock_file.seek(0)
                    lock_file.truncate()
                    lock_file.write(f"pid={os.getpid()} db_path={self.db_path}\n")
                    lock_file.flush()
                self._process_lock_file = lock_file
                print(
                    f"[Cache] Acquired {lock_mode} process lock: "
                    f"{lock_path} (pid={os.getpid()})"
                )
                return
            except BlockingIOError:
                lock_file.seek(0)
                holder = lock_file.read().strip() or "active shared/exclusive holder"
                if timeout <= 0 or time.monotonic() >= deadline:
                    lock_file.close()
                    raise RuntimeError(
                        f"Cache DB lock is unavailable: {self.db_path} ({holder})"
                    )
                await asyncio.sleep(min(0.25, max(0.01, deadline - time.monotonic())))

    def _release_process_lock(self):
        """Release the process lock held for this cache DB."""
        if self._process_lock_file is None:
            return
        try:
            fcntl.flock(self._process_lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            self._process_lock_file.close()
            self._process_lock_file = None

    async def _initialize_tables(self):
        """Initialize cache tables."""
        for table_name, schema in CACHE_TABLES.items():
            await self._connection.executescript(schema)
        await self._connection.commit()

    async def execute_read(self, sql: str, params: tuple = ()) -> Optional[tuple]:
        """Execute a read operation (no lock needed; WAL supports concurrent reads)."""
        conn = await self.get_connection()
        async with conn.execute(sql, params) as cursor:
            return await cursor.fetchone()

    async def execute_write(self, sql: str, params: tuple = ()):
        """Execute a write operation (protected by a lock)."""
        async with self._write_lock:
            conn = await self.get_connection()
            await conn.execute(sql, params)
            await conn.commit()

    async def execute_write_batch(self, operations: list[tuple[str, tuple]]):
        """Execute batched writes (single lock acquisition for efficiency)."""
        async with self._write_lock:
            conn = await self.get_connection()
            for sql, params in operations:
                await conn.execute(sql, params)
            await conn.commit()

    async def close(self):
        """Close the database connection."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
        self._release_process_lock()


async def get_async_cache_db() -> AsyncCacheDB:
    """Get the global async database instance (thread-safe singleton)."""
    global _async_db_pool
    if _async_db_pool is None:
        async with _async_db_lock:
            if _async_db_pool is None:
                print(f"[Cache] Opening database: {CACHE_CONFIG['db_path']} (pid={os.getpid()}, CACHE_DB_PATH={os.getenv('CACHE_DB_PATH', '<not set>')})")
                _async_db_pool = AsyncCacheDB(CACHE_CONFIG["db_path"])
    return _async_db_pool


# Cache table schemas
CACHE_TABLES = {
    "text_search": """
        CREATE TABLE IF NOT EXISTS text_search (
            query_hash TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at REAL NOT NULL,
            last_accessed REAL NOT NULL,
            access_count INTEGER DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_text_search_query_hash ON text_search(query_hash);
        CREATE INDEX IF NOT EXISTS idx_text_search_last_accessed ON text_search(last_accessed);
    """,
    "text_visit": """
        CREATE TABLE IF NOT EXISTS text_visit (
            url_hash TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at REAL NOT NULL,
            last_accessed REAL NOT NULL,
            access_count INTEGER DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_text_visit_url_hash ON text_visit(url_hash);
        CREATE INDEX IF NOT EXISTS idx_text_visit_last_accessed ON text_visit(last_accessed);
    """,
    "image_search": """
        CREATE TABLE IF NOT EXISTS image_search (
            image_url_hash TEXT PRIMARY KEY,
            image_url TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at REAL NOT NULL,
            last_accessed REAL NOT NULL,
            access_count INTEGER DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_image_search_url_hash ON image_search(image_url_hash);
        CREATE INDEX IF NOT EXISTS idx_image_search_last_accessed ON image_search(last_accessed);
    """,
    "image_visit": """
        CREATE TABLE IF NOT EXISTS image_visit (
            url_hash TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at REAL NOT NULL,
            last_accessed REAL NOT NULL,
            access_count INTEGER DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_image_visit_url_hash ON image_visit(url_hash);
        CREATE INDEX IF NOT EXISTS idx_image_visit_last_accessed ON image_visit(last_accessed);
    """,
    "region_search": """
        CREATE TABLE IF NOT EXISTS region_search (
            image_hash TEXT PRIMARY KEY,
            image_id TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at REAL NOT NULL,
            last_accessed REAL NOT NULL,
            access_count INTEGER DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_region_search_image_hash ON region_search(image_hash);
        CREATE INDEX IF NOT EXISTS idx_region_search_last_accessed ON region_search(last_accessed);
    """,
    "thumbnail_image": """
        CREATE TABLE IF NOT EXISTS thumbnail_image (
            url_hash TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            image_data BLOB NOT NULL,
            created_at REAL NOT NULL,
            last_accessed REAL NOT NULL,
            access_count INTEGER DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_thumbnail_image_url_hash ON thumbnail_image(url_hash);
    """,
}


def get_cache_key(text: str) -> str:
    """Generate a cache key from text using SHA256."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ============================================================================
# Async cache operations (high-concurrency optimized)
# ============================================================================

# Column name mapping (for table-specific queries)
_HASH_COL_MAPPING = {
    "text_search": "query_hash",
    "text_visit": "url_hash",
    "image_search": "image_url_hash",
    "image_visit": "url_hash",
    "region_search": "image_hash",
    "thumbnail_image": "url_hash",
}

_INPUT_COL_MAPPING = {
    "text_search": ("query_hash", "query"),
    "text_visit": ("url_hash", "url"),
    "image_search": ("image_url_hash", "image_url"),
    "image_visit": ("url_hash", "url"),
    "region_search": ("image_hash", "image_id"),
    "thumbnail_image": ("url_hash", "url"),
}


async def get_cache_async(
    table: str, key: str, executor: ThreadPoolExecutor | None = None
) -> Optional[str]:
    """
    Fetch a cache entry asynchronously.

    High-concurrency optimizations:
    - Reads do not need a lock (WAL supports concurrent reads)
    - Use asyncio.sleep for non-blocking retries
    - The executor argument is kept for backward compatibility but unused
    """
    max_retries = CACHE_CONFIG["max_retries"]
    base_delay = CACHE_CONFIG["base_retry_delay"]
    hash_col = _HASH_COL_MAPPING.get(table, "hash")

    for attempt in range(max_retries):
        try:
            cache_db = await get_async_cache_db()
            current_time = time.time()

            # Read first (no lock needed).
            row = await cache_db.execute_read(
                f"SELECT result FROM {table} WHERE {hash_col} = ?", (key,)
            )

            if row:
                # Update access time asynchronously (write lock).
                # Fire-and-forget; do not wait for completion.
                asyncio.create_task(
                    _update_access_time_async(table, hash_col, key, current_time)
                )
                return row[0]
            else:
                return None

        except Exception as e:
            error_msg = str(e).lower()
            if (
                "database is locked" in error_msg or "database is busy" in error_msg
            ) and attempt < max_retries - 1:
                # Async wait; do not block the event loop.
                wait_time = (2**attempt) * base_delay
                await asyncio.sleep(wait_time)
                continue
            else:
                log_tool_event(
                    "Cache",
                    "AsyncGetError",
                    f"table={table} error={str(e)}",
                    level="ERROR",
                )
                return None

    return None


async def _update_access_time_async(
    table: str, hash_col: str, key: str, current_time: float
):
    """Update cache access time in the background (does not block main flow)."""
    try:
        cache_db = await get_async_cache_db()
        await cache_db.execute_write(
            f"UPDATE {table} SET last_accessed = ?, access_count = access_count + 1 WHERE {hash_col} = ?",
            (current_time, key),
        )
    except Exception:
        pass  # Ignore update failures; does not affect main flow.


async def set_cache_async(
    table: str,
    key: str,
    original_input: str,
    result: str,
    executor: ThreadPoolExecutor | None = None,
):
    """
    Store a cache entry asynchronously.

    High-concurrency optimizations:
    - Protect writes with a lock to avoid conflicts
    - Use asyncio.sleep for non-blocking retries
    - The executor argument is kept for backward compatibility but unused
    """
    max_retries = CACHE_CONFIG["max_retries"]
    base_delay = CACHE_CONFIG["base_retry_delay"]
    hash_col, input_col = _INPUT_COL_MAPPING.get(table, ("hash", "input"))

    # Validate data size.
    if len(result) > 100 * 1024 * 1024:  # 100MB limit
        log_tool_event(
            "Cache",
            "SizeError",
            f"table={table} result too large: {len(result)} bytes",
            level="WARNING",
        )
        return

    for attempt in range(max_retries):
        try:
            cache_db = await get_async_cache_db()
            current_time = time.time()

            # Write operation (protected by a lock).
            await cache_db.execute_write(
                f"""
                INSERT OR REPLACE INTO {table}
                ({hash_col}, {input_col}, result, created_at, last_accessed, access_count)
                VALUES (?, ?, ?, ?, ?, 1)
                """,
                (key, original_input, result, current_time, current_time),
            )
            return  # Success

        except Exception as e:
            error_msg = str(e).lower()
            if (
                "database is locked" in error_msg or "database is busy" in error_msg
            ) and attempt < max_retries - 1:
                # Async wait; do not block the event loop.
                wait_time = (2**attempt) * base_delay
                await asyncio.sleep(wait_time)
                continue
            else:
                log_tool_event(
                    "Cache",
                    "AsyncSetError",
                    f"table={table} error={str(e)}",
                    level="ERROR",
                )
                return


# ============================================================================
# Region search cache helpers (box-level, keyed by image content hash)
# ============================================================================


def compute_image_hash(image_path: str) -> str:
    """Compute SHA256 hash of an image file's content (path-independent)."""
    h = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


async def get_region_cache_async(image_hash: str) -> list[dict] | None:
    """Get all cached boxes for an image from the region_search table.

    Returns a list of box dicts [{bbox, search_results, search_status}, ...]
    or None if no cache exists.
    """
    raw = await get_cache_async("region_search", image_hash)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
        return data.get("boxes", [])
    except (json.JSONDecodeError, AttributeError):
        return None


async def append_region_cache_async(
    image_hash: str, image_id: str, new_box: dict
) -> None:
    """Append a new box entry to the region search cache for an image.

    Performs a read-modify-write. In rare concurrent-write scenarios a box
    entry may be lost, but this is harmless — the next query will simply
    re-execute the online search and re-cache the result.
    """
    # Read existing boxes
    existing_boxes = await get_region_cache_async(image_hash)
    boxes = existing_boxes if existing_boxes is not None else []

    # Append new box
    boxes.append(new_box)

    # Write back
    result = json.dumps({"boxes": boxes}, ensure_ascii=False)
    await set_cache_async("region_search", image_hash, image_id, result)


async def get_thumbnail_cache_async(url: str) -> bytes | None:
    """Get cached thumbnail image bytes by URL."""
    key = get_cache_key(url)
    cache_db = await get_async_cache_db()
    try:
        row = await cache_db.execute_read(
            "SELECT image_data FROM thumbnail_image WHERE url_hash = ?", (key,)
        )
        if row:
            asyncio.create_task(
                _update_access_time_async("thumbnail_image", "url_hash", key, time.time())
            )
            return row[0]
        return None
    except Exception as e:
        log_tool_event("Cache", "ThumbnailGetError", f"url={url[:80]} error={e}", level="ERROR")
        return None


async def set_thumbnail_cache_async(url: str, image_bytes: bytes) -> None:
    """Cache thumbnail image bytes by URL."""
    key = get_cache_key(url)
    if len(image_bytes) > 10 * 1024 * 1024:  # 10MB sanity limit
        return
    cache_db = await get_async_cache_db()
    current_time = time.time()
    try:
        await cache_db.execute_write(
            """
            INSERT OR REPLACE INTO thumbnail_image
            (url_hash, url, image_data, created_at, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, 1)
            """,
            (key, url, image_bytes, current_time, current_time),
        )
    except Exception as e:
        log_tool_event("Cache", "ThumbnailSetError", f"url={url[:80]} error={e}", level="ERROR")


def log_tool_event(
    source: str,
    status: str,
    message: str | None,
    *,
    error: str | None = None,
    level: str | None = "INFO",
) -> None:
    """Unified logging helper for DeepResearch tools (stdout based)."""

    safe_message = message or ""
    level_name = _normalize_level(level)

    if level_name == "DEBUG":
        return

    message_preview = shorten_for_log(safe_message)

    log_parts = [
        f"[Tool][{source}][{status}][{level_name}]",
        f"message_len={len(safe_message)}",
        f"preview={json.dumps(message_preview, ensure_ascii=False)}",
    ]

    if error is not None:
        error_preview = shorten_for_log(error)
        log_parts.append(f"error_len={len(error)}")
        log_parts.append(f"error={json.dumps(error_preview, ensure_ascii=False)}")

    print(" ".join(log_parts))


def log_search(
    source: str,
    status: str,
    query: str,
    result: str | None = None,
    error: str | None = None,
) -> None:
    """Standardized debug logs for search tools."""

    parts = [f"query={json.dumps(query, ensure_ascii=False)}"]

    if result is not None:
        preview = shorten_for_log(result)
        parts.append(f"result_len={len(result)}")
        parts.append(f"preview={json.dumps(preview, ensure_ascii=False)}")

    message = " ".join(parts)
    level = "ERROR" if error else "INFO"

    log_tool_event(
        source=f"Search/{source}",
        status=status,
        message=message,
        error=error,
        level=level,
    )


class DeepResearchTool(RLLMTool, ABC):
    """
    Base class for all DeepResearch tools.

    Inherits from rLLM's Tool to support OpenAI native function calling,
    while maintaining compatibility with ReAct text format.
    """

    def __init__(self, name: str, description: str, parameters: dict | None = None):
        """
        Initialize DeepResearch tool with OpenAI function calling support.

        Args:
            name: Tool name
            description: Tool description
            parameters: OpenAI-style parameter schema (optional)
        """
        # Set _json BEFORE calling super().__init__
        # because the parent's __init__ may access self.json
        self._json = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
                or {"type": "object", "properties": {}, "required": []},
            },
        }

        super().__init__(name=name, description=description)
        self.executor: ThreadPoolExecutor | None = None

    @abstractmethod
    async def call(self, **kwargs) -> Union[str, "ToolCallResult"]:
        """Execute the tool with given arguments."""
        pass

    def set_executor(self, executor: ThreadPoolExecutor | None) -> None:
        """Bind a tool executor for blocking calls."""
        self.executor = executor

    def _get_requests_proxies(self) -> dict | None:
        """Build requests-compatible proxy mapping from TOOL_HTTPS_PROXY."""
        proxy_value = os.getenv("TOOL_HTTPS_PROXY")
        if proxy_value is None:
            return None

        proxy_value = proxy_value.strip()
        if not proxy_value or proxy_value.lower() == "none":
            return {"http": None, "https": None}

        return {"http": proxy_value, "https": proxy_value}

    async def _run_blocking(self, func: Callable[[], T]) -> T:
        """Run a blocking function in the bound executor."""
        return await run_blocking(func, executor=self.executor)

    async def async_forward(self, **kwargs):
        """rLLM Tool interface - delegates to call()"""
        try:
            from rllm.tools.tool_base import ToolOutput
        except ImportError:
            from rllm_mllm.rllm.tools.tool_base import ToolOutput

        try:
            result = await self.call(**kwargs)
            output = result.text if isinstance(result, ToolCallResult) else result
            return ToolOutput(name=self.name, output=output)
        except Exception as e:
            return ToolOutput(name=self.name, error=f"{type(e).__name__} - {str(e)}")


async def check_cache_health_async() -> bool:
    """Check cache database health asynchronously."""
    try:
        cache_db = await get_async_cache_db()
        conn = await cache_db.get_connection()

        # Test basic connectivity
        async with conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ) as cursor:
            row = await cursor.fetchone()
            table_count = row[0] if row else 0

        # Check if our tables exist
        expected_tables = {"text_search", "text_visit", "image_search", "image_visit"}
        async with conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ) as cursor:
            rows = await cursor.fetchall()
            existing_tables = {row[0] for row in rows}

        if not expected_tables.issubset(existing_tables):
            # Tables will be auto-created by AsyncCacheDB
            pass

        # Test WAL file size (rough check)
        try:
            wal_path = CACHE_CONFIG["db_path"] + "-wal"
            if os.path.exists(wal_path):
                wal_size = os.path.getsize(wal_path)
                if wal_size > 100 * 1024 * 1024:  # 100MB
                    log_tool_event(
                        "Cache",
                        "WALSize",
                        f"WAL file too large: {wal_size} bytes",
                        level="WARNING",
                    )
        except:
            pass

        return True

    except Exception as e:
        log_tool_event(
            "Cache", "HealthCheck", f"Health check failed: {str(e)}", level="ERROR"
        )
        return False


async def cleanup_expired_cache_async():
    """Clean expired cache entries asynchronously."""
    try:
        cache_db = await get_async_cache_db()
        max_age_seconds = CACHE_CONFIG["max_age_days"] * 24 * 60 * 60
        cutoff_time = time.time() - max_age_seconds

        # Clean up expired entries
        tables = ["text_search", "text_visit", "image_search", "image_visit", "region_search", "thumbnail_image"]

        for table in tables:
            await cache_db.execute_write(
                f"DELETE FROM {table} WHERE last_accessed < ?", (cutoff_time,)
            )

    except Exception as e:
        log_tool_event(
            "Cache", "CleanupError", f"Failed to cleanup cache: {str(e)}", level="ERROR"
        )


async def initialize_cache_async():
    """Initialize the cache database asynchronously (auto on first use)."""
    try:
        cache_db = await get_async_cache_db()
        await cache_db.get_connection()  # Trigger connection and table initialization.
        await check_cache_health_async()
        await cleanup_expired_cache_async()
    except Exception as e:
        log_tool_event(
            "Cache",
            "InitError",
            f"Failed to initialize cache: {str(e)}",
            level="WARNING",
        )


# Cache initialization flag
_cache_initialized = False
_cache_init_lock = asyncio.Lock()


async def ensure_cache_initialized():
    """Ensure the cache is initialized (thread-safe)."""
    global _cache_initialized
    if not _cache_initialized:
        async with _cache_init_lock:
            if not _cache_initialized:
                await initialize_cache_async()
                _cache_initialized = True
