"""
importlib实现
加载各个模块下的plugin.yml，然后读入func_map
默认使用gemini接口标准，使用convert_gemini_to_openai将最终gemini func_map转换为openai func_map
"""

import copy
import json


def gemini_func_map():
    with open('plugins/core/gemini_func_call.json', 'r',encoding='utf-8') as f:
        data = json.load(f)
    tools = data
    return tools
def convert_gemini_to_openai(gemini_tools):
    openai_functions = []

    for tool in gemini_tools["function_declarations"]:
        print(tool)
        openai_function = {
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": copy.deepcopy(tool.get("parameters", {}))
            }
        }

        # Ensure 'parameters' has all required fields for OpenAI format
        if "parameters" in openai_function["function"]:
            parameters = openai_function["function"]["parameters"]
            parameters.setdefault("type", "object")
            parameters.setdefault("properties", {})
            parameters.setdefault("required", [])
            parameters["additionalProperties"] = False

        openai_functions.append(openai_function)
    with open('plugins/core/openai_func_call.json', 'w',encoding='utf-8') as f:

        f.write(json.dumps(openai_functions))
    return openai_functions
