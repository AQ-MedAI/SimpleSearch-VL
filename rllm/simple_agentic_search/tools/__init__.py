from simple_agentic_search.tools.region_search_tool import RegionSearchTool
from simple_agentic_search.tools.search_tool import SearchTool
from simple_agentic_search.tools.shared import DeepResearchTool
from simple_agentic_search.tools.visit_tool import VisitTool
from simple_agentic_search.tools.zoom_in_tool import ZoomInTool

# Registry of tools exposed by the current YAML configs.
# Each entry maps a tool name to (ToolClass, accepts_tools_config).
_TOOL_REGISTRY = {
    "text_search": (SearchTool, True),
    "visit": (VisitTool, True),
    "image_search": (RegionSearchTool, True),
    "zoom_in": (ZoomInTool, True),
}


def get_tool(name: str) -> DeepResearchTool | None:
    entry = _TOOL_REGISTRY.get(name)
    if entry is None:
        return None
    tool_cls, _ = entry
    return tool_cls()


def get_all_tools(tools_config: dict | None = None) -> dict[str, DeepResearchTool]:
    """Instantiate tools based on tools_config.enabled_tools."""
    cfg = dict(tools_config) if tools_config else {}
    enabled = cfg.get("enabled_tools", ["text_search", "image_search", "zoom_in", "visit"])

    tools = {}
    for name in enabled:
        if name not in _TOOL_REGISTRY:
            continue
        tool_cls, accepts_config = _TOOL_REGISTRY[name]
        if accepts_config and cfg:
            tools[name] = tool_cls(tools_config=cfg)
        else:
            tools[name] = tool_cls()

    return tools


__all__ = [
    "DeepResearchTool",
    "SearchTool",
    "VisitTool",
    "ZoomInTool",
    "RegionSearchTool",
    "_TOOL_REGISTRY",
    "get_tool",
    "get_all_tools",
]
