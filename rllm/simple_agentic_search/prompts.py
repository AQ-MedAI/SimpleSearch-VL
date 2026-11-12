"""
Prompt definitions for Mars Agentic Search — Qwen3-VL native tool-calling format.

Format:
- Tool schemas as OpenAI function-calling JSON inside <tools></tools> XML tags
- Tool calls use <tool_call></tool_call> XML tags
- Tool responses use <tool_response></tool_response> XML tags
- Reasoning uses <thinking></thinking> XML tags

System prompt is built dynamically based on enabled_tools config.
"""

import copy
import json
from datetime import datetime

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "text_search",
        "description": (
            "Perform text-based web searches and return top results. "
            "Use when you already have specific keywords or facts to look up. "
            "Supply complementary queries in a single call when useful. "
            "Call as {\"name\": \"text_search\", \"arguments\": {\"query\": [\"...\"]}}; "
            "the query array must stay inside arguments."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 3,
                    "description": "Array of search query strings.",
                },
            },
            "required": ["query"],
        },
    },
}

IMAGE_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "image_search",
        "description": (
            "Search visually similar images for one or more input image regions. "
            "Use this to identify unfamiliar objects, logos, landmarks, artworks, products, people, or scenes. "
            "Input images are referred to by img_idx, starting from 0. "
            "Each region must pair one img_idx with one bbox_2d in 0-1000 scale. "
            "Use [0, 0, 1000, 1000] for a whole image. "
            "The tool response will label each searched region with the same img_idx so you can connect results back to the input image."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "regions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
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
                    "description": (
                        'Regions to search, e.g. [{"img_idx": 0, '
                        '"bbox_2d": [0, 0, 1000, 1000]}].'
                    ),
                },
            },
            "required": ["regions"],
        },
    },
}

ZOOM_IN_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "zoom_in",
        "description": (
            "Crop one or more input image regions for closer inspection. "
            "Input images are referred to by img_idx, starting from 0. "
            "Each region must pair one img_idx with one bbox_2d in 0-1000 scale. "
            "Use [0, 0, 1000, 1000] for a whole image."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "regions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
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
                    "description": (
                        'Regions to crop, e.g. [{"img_idx": 0, '
                        '"bbox_2d": [0, 0, 1000, 1000]}].'
                    ),
                },
            },
            "required": ["regions"],
        },
    },
}

VISIT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "visit",
        "description": (
            "Visit webpages and extract evidence relevant to your goal. "
            "Returns extracted evidence, a summary, and related links for further research. "
            "Use exact, complete URLs from search results or related links. "
            "Do NOT modify, truncate, or guess URLs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 3,
                    "description": "Array of the URL of the webpages.",
                },
                "goal": {
                    "type": "string",
                    "description": "What information you want to extract from the webpages.",
                },
            },
            "required": ["url", "goal"],
        },
    },
}

# ---------------------------------------------------------------------------
# Tool schema registry — maps tool name to schema
# ---------------------------------------------------------------------------

TOOL_SCHEMA_REGISTRY = {
    "text_search": SEARCH_TOOL_SCHEMA,
    "image_search": IMAGE_SEARCH_TOOL_SCHEMA,
    "zoom_in": ZOOM_IN_TOOL_SCHEMA,
    "visit": VISIT_TOOL_SCHEMA,
}

# ---------------------------------------------------------------------------
# Legacy tool sets (kept for backward compatibility)
# ---------------------------------------------------------------------------

TEXT_TOOLS = [SEARCH_TOOL_SCHEMA, VISIT_TOOL_SCHEMA]
VISION_TOOLS = [IMAGE_SEARCH_TOOL_SCHEMA, ZOOM_IN_TOOL_SCHEMA, SEARCH_TOOL_SCHEMA, VISIT_TOOL_SCHEMA]

# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PREAMBLE = """\
#Role
You are a visual research assistant.
Given a question, your task is to solve the problem **one substep at a time**.

## Guiding Principles
At each turn, you must **either**:
1. Issue **one specific tool** enclosed in <tool_call></tool_call> tags,
2. Or provide the **final answer** enclosed in <answer></answer> tags.

All outputs **must begin with a thought** enclosed in <thinking></thinking> tags, explaining your current reasoning and what to do next.

## Output Format (strict):
Always start with <thinking>. Do not output the previous reasoning chain. Then, depending on the case, output one of the following:

1. If reasoning continues:
<thinking>Your current reasoning and next plan</thinking>
<tool_call>One precise tool call to assist your reasoning</tool_call>

2. If ready to conclude:
<thinking>Summarize all reasoning and derive the answer</thinking>
<answer>Final answer</answer>"""

_TOOLS_SECTION_TEMPLATE = """

# Tools

You may call exactly one function per assistant turn. Use batched arguments inside that single tool call when you need parallel search.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_json_lines}
</tools>

For the function call, return one valid JSON object inside <tool_call></tool_call>.
The top-level object must have exactly two sibling keys: "name" and "arguments".
Never insert an extra object between "name" and "arguments".

Use these exact JSON shapes:
<tool_call>
{{"name": "text_search", "arguments": {{"query": ["example query"]}}}}
</tool_call>
<tool_call>
{{"name": "visit", "arguments": {{"url": ["https://example.com"], "goal": "find relevant evidence"}}}}
</tool_call>
<tool_call>
{{"name": "image_search", "arguments": {{"regions": [{{"img_idx": 0, "bbox_2d": [0, 0, 1000, 1000]}}]}}}}
</tool_call>"""


def _schema_for_qwen_prompt(schema: dict) -> dict:
    rendered = copy.deepcopy(schema)
    function = rendered.get("function", {})
    if "parameters" in function and "arguments" not in function:
        function["arguments"] = function.pop("parameters")
    return rendered


def _build_tools_section(tool_schemas: list[dict]) -> str:
    lines = "\n".join(
        json.dumps(_schema_for_qwen_prompt(t), ensure_ascii=False)
        for t in tool_schemas
    )
    return _TOOLS_SECTION_TEMPLATE.format(tool_json_lines=lines)


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _configured_tool_schema(schema: dict, tools_config: dict | None) -> dict:
    cfg = _as_dict(tools_config)
    image_cfg = _as_dict(cfg.get("image_search"))
    zoom_cfg = _as_dict(cfg.get("zoom_in"))
    configured = copy.deepcopy(schema)
    function = configured.get("function", {})
    name = function.get("name")
    properties = function.get("parameters", {}).get("properties", {})

    if name == "text_search" and "query" in properties:
        max_queries = int(cfg.get("search_max_queries", properties["query"].get("maxItems", 3)))
        properties["query"]["maxItems"] = max_queries
        properties["query"]["description"] = "Array of search query strings."
        # maxItems carries the configured query limit; keep prose stable.
    elif name == "visit" and "url" in properties:
        max_urls = int(cfg.get("visit_max_urls", properties["url"].get("maxItems", 3)))
        properties["url"]["maxItems"] = max_urls
        properties["url"]["description"] = "Array of webpage URLs to visit."
    elif name == "image_search" and "regions" in properties:
        max_regions = int(image_cfg.get("max_bboxes", cfg.get("max_bboxes", properties["regions"].get("maxItems", 3))))
        properties["regions"]["maxItems"] = max_regions
    elif name == "zoom_in" and "regions" in properties:
        max_boxes = int(zoom_cfg.get("max_bboxes", cfg.get("max_bboxes", properties["regions"].get("maxItems", 3))))
        properties["regions"]["maxItems"] = max_boxes

    return configured


def build_system_prompt(
    enabled_tools: list[str] | None = None,
    has_image: bool = True,
    mode: str | None = None,
    tools_config: dict | None = None,
) -> str:
    """Build the full system prompt for the given tool set.

    Args:
        enabled_tools: List of tool names to include (e.g. ["text_search", "image_search", "visit"]).
                       If provided, only schemas for these tools are included.
        has_image: Whether the current task includes an image.
        mode: Legacy parameter. "text" or "vision". Ignored when enabled_tools is provided.

    Returns:
        Complete system prompt string.
        tools_config: YAML tools_config used to keep prompt schemas aligned with runtime limits.
    """
    # Preferred tool ordering: vision tools first for image tasks
    _VISION_TOOL_ORDER = ["image_search", "zoom_in", "text_search", "visit"]

    if enabled_tools is not None:
        # Dynamic mode: build schema list from enabled_tools
        # If no image, filter out vision-only tools before schema lookup
        vision_only_tools = {"image_search", "zoom_in"}
        active_tools = enabled_tools if has_image else [
            t for t in enabled_tools if t not in vision_only_tools
        ]
        # Sort: vision tools first when has_image
        if has_image:
            ordered = sorted(
                active_tools,
                key=lambda t: _VISION_TOOL_ORDER.index(t) if t in _VISION_TOOL_ORDER else 99,
            )
        else:
            ordered = active_tools
        schemas = []
        for tool_name in ordered:
            schema = TOOL_SCHEMA_REGISTRY.get(tool_name)
            if schema is not None:
                schemas.append(schema)
    else:
        # Legacy mode
        if mode == "text" or not has_image:
            schemas = TEXT_TOOLS
        else:
            schemas = VISION_TOOLS

    schemas = [_configured_tool_schema(schema, tools_config) for schema in schemas]

    date_str = datetime.now().date().strftime("%Y-%m-%d")
    return _SYSTEM_PREAMBLE + _build_tools_section(schemas) + f"\n\nCurrent date: {date_str}"


# ---------------------------------------------------------------------------
# Pre-built system prompts (cached at import time, legacy)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEXT = build_system_prompt(mode="text")
SYSTEM_PROMPT_VISION = build_system_prompt(mode="vision")

# ---------------------------------------------------------------------------
# Error / guidance messages
# ---------------------------------------------------------------------------

TOOL_RESPONSE_GUIDANCE = (
    "Reminder: Do NOT give your final answer until you have gathered sufficient "
    "evidence. Consider using different tools or batched inputs within one tool call "
    "to collect more information if needed."
)

SEARCH_RESULT_GUIDANCE = (
    "Search results only show snippets. "
    "Visit the relevant pages to get full details — you can pass multiple URLs in one visit call."
)

IMAGE_SEARCH_RESULT_GUIDANCE = (
    "Reminder: Review the titles and URLs above. Visit only pages that seem semantically relevant "
    "to the target region or question. You can pass multiple URLs in one visit call "
    "to gather information efficiently."
)

IMAGE_SEARCH_RESULT_GUIDANCE_WITH_THUMBNAILS = (
    "Reminder: Each image thumbnail corresponds to the result immediately below it. "
    "Compare the thumbnails with the queried image region first. Visit all URLs whose "
    "thumbnails and titles are visually and semantically consistent with the target. "
    "If thumbnails do not match the target region, treat those results as low-confidence "
    "and do not rely on them. You can pass multiple URLs in one visit call."
)

FORMAT_ERROR_MSG = (
    "Error: Invalid content format. "
    "Your response MUST start with <thinking>...</thinking> followed by either "
    "a <tool_call> block or an <answer> block. Please try again."
)

REPETITION_ERROR_MSG = (
    "Error: Content repetition detected. "
    "Your response MUST start with <thinking>...</thinking> followed by either "
    "a <tool_call> block or an <answer> block. Please try again."
)

FINAL_ANSWER_INSTRUCTION = (
    "Based on all the information above, please provide your best answer now "
    "in the format: <thinking>your final reasoning</thinking>\n"
    "<answer>your answer</answer>"
)

ROUND_LIMIT_INSTRUCTION_TEMPLATE = (
    "You have reached the maximum number of reasoning rounds ({max_rounds}). "
    "Based on all the information gathered so far, please provide your best "
    "final answer now in the format: <thinking>your final reasoning</thinking>\n"
    "<answer>your answer</answer>"
)

AGENT_TIMEOUT_FINAL_ANSWER_INSTRUCTION = (
    "The agent runtime budget has been reached. Stop using tools. "
    "Based on the information gathered so far, provide your best final answer "
    "now in the format: <thinking>your final reasoning</thinking>\n"
    "<answer>your answer</answer>"
)
