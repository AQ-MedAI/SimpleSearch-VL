from simple_agentic_search.reward.format_reward import compute_format_reward


def test_format_reward_accepts_object_arguments():
    messages = [
        {
            "role": "assistant",
            "content": (
                '<thinking>Search for evidence.</thinking>\n'
                '<tool_call>{"name":"text_search","arguments":{"query":["World Bank energy"]}}</tool_call>'
            ),
        }
    ]

    result = compute_format_reward(messages)

    assert result["reward"] == 0.5
    assert result["format_structural_errors"] == []


def test_format_reward_rejects_list_arguments():
    messages = [
        {
            "role": "assistant",
            "content": (
                '<thinking>Search for evidence.</thinking>\n'
                '<tool_call>{"name":"text_search","arguments":["World Bank energy", "power markets"]}</tool_call>'
            ),
        }
    ]

    result = compute_format_reward(messages)

    assert result["reward"] == 0.0
    assert result["format_structural_errors"] == [
        "Turn 1: tool_call 'arguments' must be an object"
    ]
