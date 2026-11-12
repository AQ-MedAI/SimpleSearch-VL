"""
Zoom-in tool for visual deep research.

Crops a region of the input image based on a bounding box (0-1000 coordinate scale),
adds padding, and returns the cropped PIL Image for injection into the conversation.

Based on SenseNova-MARS ImageZoomInTool pattern.
"""

import logging
import os
from typing import List, Optional

from PIL import Image

from simple_agentic_search.tools.shared import DeepResearchTool, ToolCallResult

logger = logging.getLogger(__name__)

# Default padding ratio around the crop region
DEFAULT_PADDING = [0.2, 0.2]
# Maximum padding in pixels
MAX_PADDING_PX = 600
# Minimum crop dimension in pixels
MIN_CROP_DIM = 28
# Resize factor alignment (Qwen3-VL uses patches of 16)
RESIZE_FACTOR = 32


def _smart_resize(width: int, height: int, factor: int = RESIZE_FACTOR,
                  min_pixels: int = 65536, max_pixels: int = 2097152) -> tuple[int, int]:
    """Resize dimensions to be divisible by factor while respecting pixel limits."""
    if width < factor or height < factor:
        return max(width, factor), max(height, factor)

    # Scale to fit within pixel limits
    current_pixels = width * height
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
        width = int(width * scale)
        height = int(height * scale)
    elif current_pixels < min_pixels:
        scale = (min_pixels / current_pixels) ** 0.5
        width = int(width * scale)
        height = int(height * scale)

    # Round to nearest factor
    width = max(factor, round(width / factor) * factor)
    height = max(factor, round(height / factor) * factor)
    return width, height


class ZoomInTool(DeepResearchTool):
    """Zoom into one or more image regions by cropping with padding."""

    def __init__(self, padding: Optional[List[float]] = None, tools_config: dict | None = None):
        cfg = tools_config or {}
        zoom_cfg = cfg.get("zoom_in", {}) if isinstance(cfg, dict) else {}
        self.MAX_BBOXES = int(zoom_cfg.get("max_bboxes", cfg.get("max_bboxes", 3)))
        super().__init__(
            name="zoom_in",
            description=(
                "Zoom in on one or more input image regions by cropping them based "
                "on bounding boxes. Input images are referred to by img_idx, starting from 0. "
                "Pass img_idx in regions. "
                "Returns higher-resolution views for closer inspection. "
                "Coordinates use 0-1000 scale."
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
        self.padding = padding or zoom_cfg.get("padding", DEFAULT_PADDING)

    async def call(
        self,
        bbox_2d: List[float] | List[List[float]] | None = None,
        source_image: Optional[Image.Image] = None,
        source_images: Optional[dict] = None,
        regions: Optional[list] = None,
        **kwargs,
    ) -> ToolCallResult:
        """Crop and zoom into one or more specified image/bbox regions."""
        source_images = source_images or {}
        raw_units = []
        if regions is not None:
            if not isinstance(regions, list) or not regions:
                return ToolCallResult(text="[ZoomIn Error] regions must be a non-empty list.")
            for region in regions:
                if not isinstance(region, dict):
                    continue
                display_id = str(region.get("display_image_id", region.get("image_id", "")))
                raw_units.append((display_id, source_images.get(display_id) or source_image, region.get("bbox_2d")))
        else:
            if source_image is None:
                return ToolCallResult(text="[ZoomIn Error] No source image available for zoom_in.")
            if not isinstance(bbox_2d, list) or not bbox_2d:
                return ToolCallResult(text="[ZoomIn Error] bbox_2d must be a non-empty list.")
            bbox_list = [bbox_2d] if isinstance(bbox_2d[0], (int, float)) else bbox_2d
            raw_units = [("", source_image, bbox) for bbox in bbox_list]

        warnings = []
        if len(raw_units) > self.MAX_BBOXES:
            warnings.append(
                f"[ZoomIn Warning] Received {len(raw_units)} regions; "
                f"only first {self.MAX_BBOXES} will be executed."
            )
        raw_units = raw_units[: self.MAX_BBOXES]

        texts = []
        crops = []
        for idx, (display_id, img, bbox) in enumerate(raw_units, 1):
            if img is None:
                texts.append(f"Region {idx}: [ZoomIn Error] No source image available for {display_id or 'selected region'}.")
                continue
            if img.mode != "RGB":
                img = img.convert("RGB")
            width, height = img.size
            if not isinstance(bbox, list) or len(bbox) != 4:
                texts.append(f"Region {idx}: [ZoomIn Error] Each bbox must be [x1, y1, x2, y2].")
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox]
            except (ValueError, TypeError) as e:
                texts.append(f"Region {idx}: [ZoomIn Error] Invalid bbox_2d format: {e}")
                continue

            out_of_range = [v for v in [x1, y1, x2, y2] if v < 0 or v > 1000]
            if out_of_range:
                texts.append(
                    f"Region {idx}: [ZoomIn Error] Coordinates must be in 0-1000. "
                    f"Got bbox_2d=[{x1}, {y1}, {x2}, {y2}], out_of_range={out_of_range}."
                )
                continue

            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            px_x1 = x1 * width / 1000.0
            px_y1 = y1 * height / 1000.0
            px_x2 = x2 * width / 1000.0
            px_y2 = y2 * height / 1000.0

            crop_w = px_x2 - px_x1
            crop_h = px_y2 - px_y1
            if crop_w < 1 or crop_h < 1:
                texts.append(f"Region {idx}: [ZoomIn Error] Bounding box too small.")
                continue

            pad_x = min(crop_w * self.padding[0], MAX_PADDING_PX)
            pad_y = min(crop_h * self.padding[1], MAX_PADDING_PX)

            final_x1 = max(0, int(px_x1 - pad_x))
            final_y1 = max(0, int(px_y1 - pad_y))
            final_x2 = min(width, int(px_x2 + pad_x))
            final_y2 = min(height, int(px_y2 + pad_y))

            if (final_x2 - final_x1) < MIN_CROP_DIM:
                center = (final_x1 + final_x2) / 2
                final_x1 = max(0, int(center - MIN_CROP_DIM / 2))
                final_x2 = min(width, final_x1 + MIN_CROP_DIM)
            if (final_y2 - final_y1) < MIN_CROP_DIM:
                center = (final_y1 + final_y2) / 2
                final_y1 = max(0, int(center - MIN_CROP_DIM / 2))
                final_y2 = min(height, final_y1 + MIN_CROP_DIM)

            cropped = img.crop((final_x1, final_y1, final_x2, final_y2))
            new_w, new_h = _smart_resize(
                cropped.width, cropped.height,
                factor=RESIZE_FACTOR,
                min_pixels=65536,
                max_pixels=2097152,
            )
            if (new_w, new_h) != (cropped.width, cropped.height):
                cropped = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

            region_str = f"[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]"
            label_parts = []
            if display_id:
                label_parts.append(f"img_idx={display_id.removeprefix('img_') if display_id.startswith('img_') else display_id}")
            if len(raw_units) > 1:
                label_parts.append(f"bbox={[int(x1), int(y1), int(x2), int(y2)]}")
            prefix = f"Region {idx} {' '.join(label_parts)}: " if label_parts else ""
            texts.append(
                f"{prefix}Here is the zoomed-in view of region {region_str}. "
                f"Cropped size: {cropped.width}x{cropped.height}px."
            )
            crops.append(cropped)

        if not crops and not texts:
            texts.append("[ZoomIn Error] No valid bounding boxes provided.")
        return ToolCallResult(
            text="\n\n=======\n\n".join(warnings + texts),
            images=crops,
        )

    # get_last_crop() removed — crop is now returned directly via ToolCallResult.images
