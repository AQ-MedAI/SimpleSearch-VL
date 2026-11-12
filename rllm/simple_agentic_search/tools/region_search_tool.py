"""
Region Search Tool — offline, cached, and online modes.

Offline mode:
  Matches model-generated bounding boxes to pre-computed cached boxes via IoU,
  then returns cached search results (title + link only).

Cached mode (recommended):
  Dynamic SQLite-based cache keyed by image content hash (SHA256).
  On each bbox query: check cache via IoU matching → if hit, return cached
  results; if miss, run online search (Serper Lens) and save to cache.
  Cache grows organically with usage.

Online mode:
  Crops the image region, uploads it to a public object URL, and calls Serper Lens API.
  No caching.
"""

import asyncio
import contextvars
import hashlib
import io
import json
import logging
import os
from functools import lru_cache
from typing import Optional

from simple_agentic_search.tools.shared import (
    DeepResearchTool,
    ToolCallResult,
    append_region_cache_async,
    compute_image_hash,
    get_region_cache_async,
    log_tool_event,
    run_with_retries_async,
    track_api_usage,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4096)
def _cached_image_hash(image_path: str) -> str:
    """Compute and cache image content hash (process-level LRU)."""
    return compute_image_hash(image_path)


# Per-async-task context vars — each asyncio Task (i.e. each rollout) gets
# its own copy, so concurrent rollouts never share cached_data or match_log.
_CTX_CACHED_DATA: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "region_search_cached_data", default=None
)
_CTX_MATCH_LOG: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "region_search_match_log", default=None
)


def _compute_iou(box_a: list, box_b: list) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format (same coord scale)."""
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0

    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    # Reject invalid boxes (x1 > x2 or y1 > y2)
    if x1_a > x2_a or y1_a > y2_a or x1_b > x2_b or y1_b > y2_b:
        return 0.0

    x1 = max(x1_a, x1_b)
    y1 = max(y1_a, y1_b)
    x2 = min(x2_a, x2_b)
    y2 = min(y2_a, y2_b)

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)

    if area_a == 0 or area_b == 0:
        return 0.0

    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _format_search_results(results: list[dict], max_results: int) -> str:
    """Format search results as numbered list with title + link."""
    truncated = results[:max_results]
    if not truncated:
        return (
            "Region search failed: the matched region has no search results. "
            "Try using a different bounding box or use text search instead."
        )

    lines = [f"Region search for the specified area returned {len(truncated)} results:\n"]
    for idx, item in enumerate(truncated, 1):
        title = item.get("title", "Untitled")
        link = item.get("link", "")
        if link:
            lines.append(f"{idx}. [{title}]({link})")
        else:
            lines.append(f"{idx}. {title}")

    return "\n".join(lines)


def _format_search_results_with_thumbnails(results: list[dict], max_results: int) -> str:
    """Format search results with <image> placeholders for successful thumbnails."""
    truncated = results[:max_results]
    if not truncated:
        return (
            "Region search failed: the matched region has no search results. "
            "Try using a different bounding box or use text search instead."
        )

    lines = [f"Region search for the specified area returned {len(truncated)} results:\n"]
    for idx, item in enumerate(truncated, 1):
        title = item.get("title", "Untitled")
        link = item.get("link", "")
        has_thumb = item.get("_has_thumbnail", False)

        if has_thumb:
            lines.append("<image>")
        if link:
            lines.append(f"{idx}. [{title}]({link})")
        else:
            lines.append(f"{idx}. {title}")
        lines.append("")

    return "\n".join(lines).rstrip()


_OFFLINE_FAILURE_MSG = (
    "Region search failed: no matching results found for the specified region. "
    "Try adjusting the bounding box or use text search instead."
)

_NO_CACHE_MSG = (
    "Region search failed: no cached search data available for this sample. "
    "Try using text search instead."
)


class RegionSearchTool(DeepResearchTool):
    """Region-based image search tool with offline cache and online API modes.

    Concurrency-safe: per-sample context is stored in contextvars so that
    concurrent asyncio rollouts sharing the same tool instance are fully
    isolated (threading.local does NOT isolate coroutines on the same thread).

    No unpicklable fields on the instance, so Ray serialization works.
    """

    MAX_BBOXES = 3

    def __init__(self, tools_config: dict | None = None):
        cfg = tools_config or {}
        region_cfg = cfg.get("image_search", {})
        self.MAX_BBOXES = int(region_cfg.get("max_bboxes", cfg.get("max_bboxes", self.MAX_BBOXES)))

        super().__init__(
            name="image_search",
            description=(
                "Search the web using cropped regions of the image to find out "
                "what they are. Finds visually similar images online and returns "
                "their associated text descriptions, titles, links, and sometimes thumbnails. "
                "When thumbnails are returned, use them to judge visual match confidence before visiting URLs. "
                "Input images are referred to by img_idx, starting from 0. "
                "Pass img_idx in regions and read the same img_idx back in Region labels. "
                "Use whenever you cannot recognize something in the image — "
                "such as unfamiliar objects, products, brands, logos, landmarks, "
                "buildings, artwork, species, or any element whose identity is unclear."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "regions": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": self.MAX_BBOXES,
                        "items": {
                            "type": "object",
                            "properties": {
                                "img_idx": {
                                    "type": "number",
                                    "description": "Index of the input image, starting from 0.",
                                },
                                "bbox_2d": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 4,
                                    "maxItems": 4,
                                    "description": "Bounding box [x1, y1, x2, y2] in 0-1000 scale.",
                                },
                            },
                            "required": ["img_idx", "bbox_2d"],
                        },
                    },
                },
                "required": ["regions"],
            },
        )

        self.mode = region_cfg.get("mode", "offline")
        self.use_cache = region_cfg.get("use_cache", True)
        self.cache_root_dir = region_cfg.get("cache_root_dir", "")
        self.cache_dataset_mapping = region_cfg.get("cache_dataset_mapping", {})
        self.iou_threshold = float(region_cfg.get("iou_threshold", 0.15))
        # IoU threshold for dynamic cache matching (higher = stricter)
        self.cached_iou_threshold = float(region_cfg.get("cached_iou_threshold", 0.8))
        self.max_search_results = int(region_cfg.get("max_search_results", 5))
        self.use_thumbnails = region_cfg.get("use_thumbnails", False)
        self.thumbnail_online_download = region_cfg.get("thumbnail_online_download", False)
        self.thumbnail_download_timeout = float(region_cfg.get("thumbnail_download_timeout", 10.0))
        self.thumbnail_max_bytes = int(region_cfg.get("thumbnail_max_bytes", 5 * 1024 * 1024))
        self.thumbnail_trust_env = bool(region_cfg.get("thumbnail_trust_env", False))
        self.api_max_retries = int(cfg.get("api_max_retries", 3))

        # Online mode: read API keys from env
        self.serp_api_key = os.getenv("SERP_API_KEY", "")
        self.zhipu_api_key = os.getenv("ZHIPU_API_KEY", "")

    # ------------------------------------------------------------------
    # Context-var accessors (isolated per asyncio Task = per rollout)
    # ------------------------------------------------------------------
    def _get_cached_data(self) -> dict | None:
        return _CTX_CACHED_DATA.get()

    def _set_cached_data(self, data: dict | None) -> None:
        _CTX_CACHED_DATA.set(data)

    def _get_match_log(self) -> list:
        log = _CTX_MATCH_LOG.get()
        if log is None:
            log = []
            _CTX_MATCH_LOG.set(log)
        return log

    def get_match_log(self) -> list[dict]:
        """Return the bbox match log for the current sample.

        Each entry: {"predicted_bbox": [...], "matched_bbox": [...] | None,
                     "iou": float, "matched": bool}
        Called by the workflow after agent.run() to save into rollout data.
        """
        return list(self._get_match_log())

    # ------------------------------------------------------------------
    # Thumbnail fetching
    # ------------------------------------------------------------------
    async def _fetch_single_thumbnail(self, url: str) -> bytes | None:
        """Download a single thumbnail directly, with SQLite cache fallback."""
        if not url:
            return None

        from simple_agentic_search.tools.shared import (
            get_thumbnail_cache_async,
            set_thumbnail_cache_async,
        )

        cached = await get_thumbnail_cache_async(url)
        if cached is not None:
            return cached

        if not self.thumbnail_online_download:
            return None

        try:
            import httpx
            from PIL import Image as PILImage

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                ),
                "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            }
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=self.thumbnail_download_timeout,
                headers=headers,
                trust_env=self.thumbnail_trust_env,
            ) as client:
                resp = await client.get(url)

            if resp.status_code != 200:
                log_tool_event(
                    "RegionSearch",
                    "ThumbnailDownloadFailed",
                    f"url={url[:80]} status={resp.status_code}",
                    level="WARNING",
                )
                return None

            image_bytes = resp.content
            if not image_bytes:
                return None
            if len(image_bytes) > self.thumbnail_max_bytes:
                log_tool_event(
                    "RegionSearch",
                    "ThumbnailTooLarge",
                    f"url={url[:80]} bytes={len(image_bytes)} limit={self.thumbnail_max_bytes}",
                    level="WARNING",
                )
                return None

            content_type = resp.headers.get("content-type", "").lower()
            if content_type and "image" not in content_type:
                log_tool_event(
                    "RegionSearch",
                    "ThumbnailNonImageResponse",
                    f"url={url[:80]} content_type={content_type[:80]}",
                    level="WARNING",
                )
                return None

            PILImage.open(io.BytesIO(image_bytes)).verify()
            await set_thumbnail_cache_async(url, image_bytes)
            return image_bytes

        except Exception as e:
            log_tool_event(
                "RegionSearch", "ThumbnailFetchError",
                f"url={url[:80]} error={type(e).__name__}: {e}",
                level="WARNING",
            )
            return None

    async def _fetch_thumbnails(self, search_results: list[dict]) -> tuple[list, list[str]]:
        """Fetch thumbnails concurrently for all search results.

        Returns (pil_images, urls) where urls[i] corresponds to pil_images[i].
        Also modifies search_results in-place: sets '_has_thumbnail' = True
        on entries whose thumbnail was successfully fetched.
        """
        from PIL import Image as PILImage

        urls = [r.get("thumbnailUrl", "") for r in search_results]

        tasks = [self._fetch_single_thumbnail(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        pil_images = []
        pil_urls = []
        for i, raw in enumerate(results):
            if isinstance(raw, bytes) and raw:
                try:
                    img = PILImage.open(io.BytesIO(raw)).convert("RGB")
                    search_results[i]["_has_thumbnail"] = True
                    pil_images.append(img)
                    pil_urls.append(urls[i])
                except Exception:
                    pass

        return pil_images, pil_urls

    # ------------------------------------------------------------------
    # Sample context management
    # ------------------------------------------------------------------
    def set_sample_context(self, data_source: str, sample_idx: int) -> None:
        """Set the current sample context and pre-load cache if in offline mode.

        Called by the workflow before each agent run. Each asyncio Task
        (rollout) gets its own context vars, so concurrent rollouts are isolated.
        """
        self._set_cached_data(None)
        _CTX_MATCH_LOG.set([])

        if self.mode != "offline":
            return

        if not self.cache_root_dir:
            logger.debug("RegionSearch: cache_root_dir empty, data_source=%s", data_source)
            return

        # Resolve cache subdirectory from mapping
        cache_subdir = self.cache_dataset_mapping.get(data_source, "")
        if not cache_subdir:
            logger.debug("RegionSearch: data_source=%s not in cache_dataset_mapping", data_source)
            return

        cache_path = os.path.join(
            self.cache_root_dir, cache_subdir, f"{sample_idx}.json"
        )

        if not os.path.isfile(cache_path):
            logger.debug("RegionSearch: cache file not found: %s", cache_path)
            return

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Cache file must be a JSON object")
            # Normalize: ensure boxes is a list
            if "boxes" in data and data["boxes"] is None:
                data["boxes"] = []
            self._set_cached_data(data)
            logger.debug(
                "RegionSearch CacheLoaded: data_source=%s sample_idx=%s boxes=%d",
                data_source, sample_idx, len(data.get('boxes', [])),
            )
        except Exception as e:
            log_tool_event(
                "RegionSearch", "CacheLoadError",
                f"path={cache_path} error={e}",
                level="ERROR",
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    async def call(
        self,
        bbox_2d: list | None = None,
        goal: str = "",
        image_id: str = "",
        regions: list | None = None,
        **kwargs,
    ) -> str:
        """Execute region search for one or more image/bbox regions."""
        warnings = []

        if regions is not None:
            if not isinstance(regions, list) or not regions:
                return "Region search error: regions must be a non-empty list."
            raw_units = []
            for idx, region in enumerate(regions, 1):
                if not isinstance(region, dict):
                    return f"Region search error: regions[{idx}] must be an object."
                raw_units.append(
                    (
                        str(region.get("image_id", "")),
                        str(region.get("display_image_id", region.get("image_id", ""))),
                        region.get("bbox_2d"),
                    )
                )
        else:
            if bbox_2d is None or not isinstance(bbox_2d, list) or len(bbox_2d) == 0:
                return "Region search error: regions must be provided."
            bbox_list = [bbox_2d] if isinstance(bbox_2d[0], (int, float)) else bbox_2d
            image_ids = image_id
            if image_ids in (None, ""):
                image_ids = [""]
            elif isinstance(image_ids, str):
                image_ids = [image_ids]
            elif isinstance(image_ids, (list, tuple)):
                image_ids = [str(item) for item in image_ids if item not in (None, "")]
            else:
                return "Region search error: image_id must be a string or a non-empty list of strings."
            if not image_ids:
                image_ids = [""]

            if len(image_ids) == 1:
                raw_units = [(image_ids[0], image_ids[0], bbox) for bbox in bbox_list]
            elif len(bbox_list) == 1:
                raw_units = [(img_id, img_id, bbox_list[0]) for img_id in image_ids]
            elif len(image_ids) == len(bbox_list):
                raw_units = [(img_id, img_id, bbox) for img_id, bbox in zip(image_ids, bbox_list)]
            else:
                return (
                    "Region search error: image_id and bbox_2d are ambiguous. "
                    "Use regions=[{img_idx, bbox_2d}, ...]."
                )

        if len(raw_units) > self.MAX_BBOXES:
            warnings.append(
                f"Region search warning: received {len(raw_units)} image/box searches; "
                f"only first {self.MAX_BBOXES} will be executed."
            )
        raw_units = raw_units[: self.MAX_BBOXES]

        search_units = []
        for target_image_id, display_image_id, bbox_raw in raw_units:
            if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
                continue
            try:
                bbox_int = [int(v) for v in bbox_raw]
            except (TypeError, ValueError):
                continue
            out_of_range = [v for v in bbox_int if v < 0 or v > 1000]
            if out_of_range:
                return (
                    "Region search error: Bounding box coordinates must be in the 0-1000 range. "
                    f"Got bbox_2d={bbox_int}, where {out_of_range} are out of range. "
                    "Please provide coordinates where all values are between 0 and 1000."
                )
            search_units.append((target_image_id, display_image_id, bbox_int))

        if not search_units:
            return "Region search error: no valid regions provided. Each bbox_2d must be [x1, y1, x2, y2]."

        logger.debug("RegionSearch Call: mode=%s units=%s", self.mode, search_units)

        async def process_bbox(target_image_id: str, bbox: list[int]) -> tuple[str, list, list]:
            if self.mode == "offline":
                return self._offline_search(bbox, goal), [], []
            if self.mode == "online" and self.use_cache:
                return await self._cached_search(bbox, goal, target_image_id)
            return await self._online_search(bbox, goal, target_image_id)

        bbox_results = await asyncio.gather(
            *(process_bbox(target_image_id, bbox) for target_image_id, _, bbox in search_units),
            return_exceptions=True,
        )

        results = []
        all_thumbnail_images = []
        all_thumbnail_urls = []
        for idx, item in enumerate(bbox_results, 1):
            _, display_image_id, bbox = search_units[idx - 1]
            if isinstance(item, Exception):
                results.append(f"Region {idx}: Region search failed with error: {item}")
                continue
            result_str, imgs, urls = item
            label_parts = []
            if display_image_id:
                label_parts.append(f"img_idx={display_image_id.removeprefix('img_') if display_image_id.startswith('img_') else display_image_id}")
            label_parts.append(f"bbox={bbox}")
            prefix = f"Region {idx} {' '.join(label_parts)}:\n"
            results.append(prefix + result_str)
            all_thumbnail_images.extend(imgs)
            all_thumbnail_urls.extend(urls)

        text_parts = warnings + results
        return ToolCallResult(
            text="\n\n=======\n\n".join(text_parts),
            images=all_thumbnail_images,
            image_urls=all_thumbnail_urls,
        )

    # ------------------------------------------------------------------
    # Offline mode: cache-based search
    # ------------------------------------------------------------------
    def _offline_search(self, bbox: list[int], goal: str) -> str:
        """Match model bbox to cached boxes via IoU, return cached search results."""
        cached_data = self._get_cached_data()
        if cached_data is None:
            return _NO_CACHE_MSG

        cached_boxes = cached_data.get("boxes", [])
        if not cached_boxes:
            return _NO_CACHE_MSG

        # Find best matching cached box by IoU
        best_iou = 0.0
        best_box_data = None

        for box_data in cached_boxes:
            cached_bbox = box_data.get("bbox", [])
            if len(cached_bbox) != 4:
                continue

            # Validate bbox values are numeric
            try:
                cached_bbox = [float(v) for v in cached_bbox]
            except (TypeError, ValueError):
                continue

            iou = _compute_iou(bbox, cached_bbox)
            if iou > best_iou:
                best_iou = iou
                best_box_data = box_data

        logger.debug(
            "RegionSearch IoUMatch: model_bbox=%s best_iou=%.3f threshold=%.2f",
            bbox, best_iou, self.iou_threshold,
        )

        # Record match info for rollout visualization
        matched = best_iou >= self.iou_threshold and best_box_data is not None
        self._get_match_log().append({
            "predicted_bbox": bbox,
            "matched_bbox": best_box_data.get("bbox") if best_box_data else None,
            "iou": round(best_iou, 4),
            "matched": matched,
        })

        if not matched:
            return _OFFLINE_FAILURE_MSG

        # Check search status
        search_status = best_box_data.get("search_status", "")
        if search_status != "success":
            return _OFFLINE_FAILURE_MSG

        search_results = best_box_data.get("search_results", [])
        if not search_results:
            return _OFFLINE_FAILURE_MSG

        return _format_search_results(search_results, self.max_search_results)

    # ------------------------------------------------------------------
    # Cached mode: SQLite cache with IoU matching + online fallback
    # ------------------------------------------------------------------
    async def _cached_search(
        self, bbox: list[int], goal: str, image_id: str
    ) -> tuple[str, list, list]:
        """Check SQLite cache first (IoU match), fall back to online search.

        Returns (result_text, pil_images, pil_urls).
        """
        if not image_id:
            return "Region search error: image_id is required for cached mode.", [], []

        # Compute image content hash (LRU-cached at process level)
        try:
            image_hash = await self._run_blocking(
                lambda: _cached_image_hash(image_id)
            )
        except Exception as e:
            log_tool_event(
                "RegionSearch", "HashError",
                f"image_id={image_id} error={e}",
                level="ERROR",
            )
            return "Region search error: failed to compute image hash for the selected input image.", [], []

        # Step 1: Check SQLite cache for this image
        cached_boxes = await get_region_cache_async(image_hash)

        if cached_boxes:
            # Find best matching cached box by IoU
            best_iou = 0.0
            best_box_data = None

            for box_data in cached_boxes:
                cached_bbox = box_data.get("bbox", [])
                if len(cached_bbox) != 4:
                    continue
                try:
                    cached_bbox = [float(v) for v in cached_bbox]
                except (TypeError, ValueError):
                    continue

                iou = _compute_iou(bbox, cached_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_box_data = box_data

            # Record match info for rollout visualization
            matched = best_iou >= self.cached_iou_threshold and best_box_data is not None
            self._get_match_log().append({
                "predicted_bbox": bbox,
                "matched_bbox": best_box_data.get("bbox") if best_box_data else None,
                "iou": round(best_iou, 4),
                "matched": matched,
                "source": "cache",
            })

            if matched:
                search_results = best_box_data.get("search_results", [])
                search_status = best_box_data.get("search_status", "")
                if search_status == "success" and search_results:
                    logger.info(
                        "RegionSearch CacheHit: bbox=%s matched=%s iou=%.3f results=%d image=%s",
                        bbox, best_box_data.get("bbox"), best_iou,
                        len(search_results), image_hash[:12],
                    )
                    if self.use_thumbnails:
                        truncated = search_results[:self.max_search_results]
                        pil_images, pil_urls = await self._fetch_thumbnails(truncated)
                        return _format_search_results_with_thumbnails(truncated, self.max_search_results), pil_images, pil_urls
                    return _format_search_results(search_results, self.max_search_results), [], []

        # Step 2: Cache miss — run online search
        logger.debug(
            "RegionSearch CacheMiss: image_hash=%s bbox=%s, running online search",
            image_hash[:12], bbox,
        )
        search_results, search_status = await self._online_search_raw(bbox, goal, image_id)

        # Step 3: Save result to cache (fire-and-forget)
        new_box = {
            "bbox": bbox,
            "search_results": search_results,
            "search_status": search_status,
        }
        try:
            await append_region_cache_async(image_hash, image_id, new_box)
        except Exception as e:
            log_tool_event(
                "RegionSearch", "CacheSaveError",
                f"image_hash={image_hash[:12]} error={e}",
                level="WARNING",
            )

        if not cached_boxes:
            # First time this image appears — record as cache miss
            self._get_match_log().append({
                "predicted_bbox": bbox,
                "matched_bbox": None,
                "iou": 0.0,
                "matched": False,
                "source": "online",
            })

        if search_status != "success" or not search_results:
            return (
                "Region search failed: online search returned no results. "
                "Try adjusting the bounding box or use text search instead."
            ), [], []

        if self.use_thumbnails:
            truncated = search_results[:self.max_search_results]
            pil_images, pil_urls = await self._fetch_thumbnails(truncated)
            return _format_search_results_with_thumbnails(truncated, self.max_search_results), pil_images, pil_urls
        return _format_search_results(search_results, self.max_search_results), [], []

    # ------------------------------------------------------------------
    # Online search (raw): returns (search_results, status) for reuse
    # ------------------------------------------------------------------
    async def _online_search_raw(
        self, bbox: list[int], goal: str, image_id: str
    ) -> tuple[list[dict], str]:
        """Crop, upload, search via Serper Lens. Returns (results_list, status_str)."""
        try:
            import requests
            from PIL import Image
        except ImportError:
            return [], "error_dependency"

        if not self.serp_api_key:
            return [], "error_no_api_key"

        # Crop image
        try:
            with Image.open(image_id) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                width, height = img.size
                x1 = max(0, min(int(bbox[0] * width / 1000), width - 1))
                y1 = max(0, min(int(bbox[1] * height / 1000), height - 1))
                x2 = max(0, min(int(bbox[2] * width / 1000), width - 1))
                y2 = max(0, min(int(bbox[3] * height / 1000), height - 1))
                if x2 <= x1 or y2 <= y1:
                    return [], "error_invalid_bbox"
                cropped = img.crop((x1, y1, x2, y2))
                cropped = cropped.resize(
                    (cropped.width * 2, cropped.height * 2),
                    Image.Resampling.LANCZOS,
                )
        except Exception as e:
            log_tool_event("RegionSearch", "CropError", str(e), level="ERROR")
            return [], "error_crop"

        # Upload to OSS
        try:
            import oss2

            oss_access_key_id = os.getenv("OSS_ACCESS_KEY_ID", "")
            oss_access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET", "")
            oss_endpoint = os.getenv("OSS_ENDPOINT", "")
            oss_bucket_name = os.getenv("OSS_BUCKET_NAME", "")
            oss_prefix_name = os.getenv("OSS_PREFIX_NAME", "")

            if not all([oss_access_key_id, oss_access_key_secret, oss_endpoint, oss_bucket_name]):
                return [], "error_oss_config"

            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cropped.save(tmp, "JPEG", quality=95)
                tmp_path = tmp.name

            try:
                image_name = os.path.splitext(os.path.basename(image_id))[0]
                bbox_str = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
                filename = f"crop_region_{image_name}_{bbox_str}.jpg"
                oss_path = os.path.join(oss_prefix_name, filename) if oss_prefix_name else filename

                auth = oss2.Auth(oss_access_key_id, oss_access_key_secret)
                bucket = oss2.Bucket(auth, oss_endpoint, oss_bucket_name)
                bucket.put_object_from_file(oss_path, tmp_path)
                oss_url = bucket.sign_url("GET", oss_path, 3600)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        except Exception as e:
            log_tool_event("RegionSearch", "UploadError", str(e), level="ERROR")
            return [], "error_upload"

        # Call Serper Lens API
        try:
            headers = {
                "X-API-KEY": self.serp_api_key,
                "Content-Type": "application/json",
            }
            payload = {"url": oss_url}
            proxies = self._get_requests_proxies()

            response = await run_with_retries_async(
                lambda: requests.post(
                    "https://google.serper.dev/lens",
                    headers=headers,
                    json=payload,
                    timeout=60,
                    proxies=proxies,
                ),
                executor=self.executor,
                attempts=self.api_max_retries,
                source="RegionSearchTool/SerperLens",
            )
            response.raise_for_status()
            result_data = response.json()

            # Track: 1 image search = 3 credits
            track_api_usage("image_search")

            search_results = []
            for item in result_data.get("organic", [])[:10]:
                link = item.get("link", "")
                if not link:
                    continue
                entry = {
                    k: v for k, v in item.items()
                    if isinstance(v, (str, int, float, bool)) and v
                }
                entry.setdefault("title", "Untitled")
                search_results.append(entry)

            status = "success" if search_results else "empty"
            return search_results, status

        except Exception as e:
            log_tool_event(
                "RegionSearch", "OnlineSearchError",
                f"error={e}",
                level="ERROR",
            )
            return [], "error_search"

    # ------------------------------------------------------------------
    # Online mode: Serper Lens API
    # ------------------------------------------------------------------
    async def _online_search(
        self, bbox: list[int], goal: str, image_id: str
    ) -> tuple[str, list, list]:
        """Crop image, upload to OSS, call Serper Lens, return (text, images, urls)."""
        try:
            import requests
            from PIL import Image
        except ImportError:
            return "Region search error: online mode requires PIL and requests packages.", [], []

        if not image_id:
            return "Region search error: image_id is required for online mode.", [], []

        if not self.serp_api_key:
            return "Region search error: SERP_API_KEY not set for online mode.", [], []

        # Crop image
        try:
            with Image.open(image_id) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                width, height = img.size
                x1 = max(0, min(int(bbox[0] * width / 1000), width - 1))
                y1 = max(0, min(int(bbox[1] * height / 1000), height - 1))
                x2 = max(0, min(int(bbox[2] * width / 1000), width - 1))
                y2 = max(0, min(int(bbox[3] * height / 1000), height - 1))
                if x2 <= x1 or y2 <= y1:
                    return "Region search error: invalid bounding box coordinates.", [], []
                cropped = img.crop((x1, y1, x2, y2))
                cropped = cropped.resize(
                    (cropped.width * 2, cropped.height * 2),
                    Image.Resampling.LANCZOS,
                )
        except Exception as e:
            return f"Region search error: failed to crop image: {e}", [], []

        # Upload to OSS
        try:
            import oss2

            oss_access_key_id = os.getenv("OSS_ACCESS_KEY_ID", "")
            oss_access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET", "")
            oss_endpoint = os.getenv("OSS_ENDPOINT", "")
            oss_bucket_name = os.getenv("OSS_BUCKET_NAME", "")
            oss_prefix_name = os.getenv("OSS_PREFIX_NAME", "")

            if not all([oss_access_key_id, oss_access_key_secret, oss_endpoint, oss_bucket_name]):
                return "Region search error: OSS configuration missing for online mode.", [], []

            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cropped.save(tmp, "JPEG", quality=95)
                tmp_path = tmp.name

            try:
                image_name = os.path.splitext(os.path.basename(image_id))[0]
                bbox_str = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
                filename = f"crop_region_{image_name}_{bbox_str}.jpg"
                oss_path = os.path.join(oss_prefix_name, filename) if oss_prefix_name else filename

                auth = oss2.Auth(oss_access_key_id, oss_access_key_secret)
                bucket = oss2.Bucket(auth, oss_endpoint, oss_bucket_name)
                bucket.put_object_from_file(oss_path, tmp_path)
                oss_url = bucket.sign_url("GET", oss_path, 3600)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        except Exception as e:
            return f"Region search error: failed to upload to OSS: {e}", [], []

        # Call Serper Lens API
        try:
            headers = {
                "X-API-KEY": self.serp_api_key,
                "Content-Type": "application/json",
            }
            payload = {"url": oss_url}
            proxies = self._get_requests_proxies()

            response = await self._run_blocking(
                lambda: requests.post(
                    "https://google.serper.dev/lens",
                    headers=headers,
                    json=payload,
                    timeout=60,
                    proxies=proxies,
                )
            )
            response.raise_for_status()
            result_data = response.json()

            # Track: 1 image search = 3 credits
            track_api_usage("image_search")

            search_results = []
            for item in result_data.get("organic", [])[:10]:
                link = item.get("link", "")
                if not link:
                    continue
                entry = {
                    k: v for k, v in item.items()
                    if isinstance(v, (str, int, float, bool)) and v
                }
                entry.setdefault("title", "Untitled")
                search_results.append(entry)

            if self.use_thumbnails:
                truncated = search_results[:self.max_search_results]
                pil_images, pil_urls = await self._fetch_thumbnails(truncated)
                return _format_search_results_with_thumbnails(truncated, self.max_search_results), pil_images, pil_urls
            return _format_search_results(search_results, self.max_search_results), [], []

        except Exception as e:
            log_tool_event(
                "RegionSearch", "OnlineSearchError",
                f"error={e}",
                level="ERROR",
            )
            return f"Region search error: online search failed: {e}", [], []
