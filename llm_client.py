"""
llm_client.py 负责和大模型 API 通信。

它的核心职责：
1. 把“用户需求”拼成提示词（prompt）。
2. 调用远程模型接口。
3. 尝试从模型返回内容里提取有效 JSON。
4. 给 app.py 提供两个稳定入口：
   - 生成问卷
   - 修改问卷
"""

# json: 处理 JSON 序列化 / 反序列化
import json
# os: 读取环境变量
import os
# re: 正则表达式，用于从文本中提取 JSON 片段
import re
# traceback: 打印详细异常堆栈，便于排查问题
import traceback
# 类型标注工具
from typing import Any, Dict, Optional

# requests: 发 HTTP 请求调用模型接口
import requests

# 模型服务接口地址。若环境变量未配置，使用默认占位地址（通常应在 .env 覆盖）。
API_URL = os.getenv("DASHSCOPE_API_URL", "https://api.dashscope.example.com/v1/chat/completions")
# 接口鉴权 Key，必须配置，否则无法请求。
API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 默认模型别名（前端不传 model 时使用）。
DEFAULT_MODEL_NAME = "deepseek-r1"

# 前端可选模型别名 -> 实际模型名称映射。
# 作用：前端用短名字，后端统一映射成真实模型 ID。
MODEL_ALIAS_MAP = {
    "deepseek-r1": "deepseek-r1",
    "qwen": "qwen-plus-2025-07-28",
    "qwen-plus-2025-07-28": "qwen-plus-2025-07-28",
}

# 单次请求超时时间（秒）。
TIMEOUT_SECONDS = 60

# 解析失败时的最大重试次数（请求重试和 JSON 提取都使用这个值）。
JSON_PARSE_RETRIES = 3


def _resolve_model_name(model: Optional[str]) -> str:
    """
    根据前端传入 model 解析最终模型名。

    规则：
    - model 为空 -> 用默认模型
    - model 在映射表中 -> 用映射后的真实模型名
    - model 不认识 -> 回退默认模型
    """
    if not model:
        return DEFAULT_MODEL_NAME
    return MODEL_ALIAS_MAP.get(model, DEFAULT_MODEL_NAME)


def _format_prompt_for_generation(user_input: str) -> str:
    """
    把“生成问卷”需求拼成给大模型的提示词。
    提示词要求模型只输出 JSON，且结构固定，减少后续解析失败。
    """
    return (
        "请根据以下用户需求生成一份问卷，输出 JSON 格式："
        "{\"title\": str, \"description\": str, \"questions\": [{\"id\": str, \"title\": str, \"type\": \"single\"|\"multiple\"|\"text\", "
        "\"options\": [str], \"required\": bool, \"depends_on\": {\"question\": str, \"option\": str}|null}]}。\n"
        "请只返回有效 JSON，不要返回多余的解释。\n"
        f"用户需求：{user_input}"
    )


def _format_prompt_for_modification(current_survey_dict: Dict[str, Any], modification: str) -> str:
    """
    把“在现有问卷上修改”的需求拼成提示词。

    注意：
    - 先把当前问卷转成 JSON 文本，明确告诉模型“当前状态是什么”。
    - 规则里强调“只改用户要求部分”，避免模型擅自改动其他题目。
    """
    # ensure_ascii=False 让中文保持可读，不转 \uXXXX。
    current = json.dumps(current_survey_dict, ensure_ascii=False)
    return (
        "你是一个问卷 JSON 编辑助手。给定当前问卷 JSON 和修改指令，输出修改后的完整问卷 JSON。\n"
        "请严格遵守以下规则：\n"
        "1. 只修改用户明确要求变更的部分，未提及字段保持不变。\n"
        "2. 输出必须是完整、可解析的 JSON 对象，不要输出解释、Markdown 或额外文本。\n"
        "3. 问卷结构固定为：\n"
        "{\"title\": str, \"description\": str, \"questions\": [{\"id\": str, \"title\": str, \"type\": \"single\"|\"multiple\"|\"text\", \"options\": [str], \"required\": bool, \"depends_on\": {\"question\": str, \"option\": str}|null}]}\n"
        "4. depends_on 的标准格式必须是 {\"question\": \"q1\", \"option\": \"选项文本\"}，不要缺少 question 或 option。\n"
        "5. 当前数据结构里，每个题目只支持一个 depends_on 对象或 null，不支持 depends_on 数组。\n"
        "6. 如果用户说“选择 A 和 B 都需要跳转”，表示要为相关后续题分别设置依赖条件；若同一题需要同时由 A 和 B 触发，由于本结构限制，应拆成两道等价题，分别设置 depends_on 为 A 和 B。\n"
        "7. 需要新增某个选项的跳转条件时，是新增对应 depends_on，不要误删其他题已有依赖。\n"
        f"当前问卷：{current}\n"
        f"修改指令：{modification}\n"
        "只返回修改后的问卷 JSON。"
    )


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    从模型返回文本中尽量提取第一个可解析的 JSON 对象（dict）。

    提取策略（从严格到宽松）：
    1. 直接整体 json.loads(text)
    2. 从 ```json ... ``` 代码块中提取
    3. 用第一对花括号做兜底提取
    """
    # 1) 优先尝试“整个文本就是 JSON”。
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2) 尝试从 Markdown 代码块提取 JSON 主体。
    matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates = matches or []

    # 3) 如果没有代码块，尝试抓取第一个大花括号片段作为候选。
    if not candidates:
        curly = re.search(r"\{.*\}", text, flags=re.S)
        if curly:
            candidates = [curly.group(0)]

    # 遍历候选，返回第一个可解析且是 dict 的 JSON。
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    return None


def _call_api(prompt: str, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    调用远程模型接口并返回问卷字典。

    返回值：
    - 成功：dict
    - 失败：None
    """
    # 没有 API_KEY 直接失败，避免发送无效请求。
    if not API_KEY:
        print("[llm_client] ERROR: DASHSCOPE_API_KEY is not configured")
        return None

    # 解析最终模型名（支持别名）。
    resolved_model = _resolve_model_name(model)

    # 组装 HTTP 请求头。
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # 组装请求体（具体格式取决于你接入的平台协议）。
    payload = {
        "model": resolved_model,
        "input": {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        },
        "parameters": {
            "result_format": "message",
            "max_tokens": 1500,
            "temperature": 0.2
        }
    }

    # 调试日志：帮助定位请求参数、模型和接口地址。
    print("[llm_client] DEBUG _call_api 请求开始")
    print(f"[llm_client] DEBUG API_URL={API_URL}")
    print(f"[llm_client] DEBUG MODEL={resolved_model}")
    print("[llm_client] DEBUG headers={'Authorization': '***', 'Content-Type': 'application/json'}")
    print(f"[llm_client] DEBUG payload={json.dumps(payload, ensure_ascii=False)[:500]}...")

    # 预留变量（当前实现中一般仍为 None；保留原逻辑不改）。
    response_text = None

    # 请求重试循环。
    for attempt in range(JSON_PARSE_RETRIES):
        try:
            print(f"[llm_client] DEBUG attempt={attempt+1}/{JSON_PARSE_RETRIES} 请求发送中...")
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)
            print(f"[llm_client] DEBUG HTTP status={resp.status_code}")
            try:
                print(f"[llm_client] DEBUG HTTP headers={dict(resp.headers)}")
            except Exception as ex:
                print(f"[llm_client] WARNING 无法读取响应头: {ex}")

            # HTTP 非 200：按失败处理，未到最大次数则重试。
            if resp.status_code != 200:
                print(f"[llm_client] ERROR 接口返回非200状态码: {resp.status_code}")
                print(f"[llm_client] ERROR 响应正文: {resp.text[:2000]}")
                # 502 等通常表示上游不稳定，可重试。
                if attempt < JSON_PARSE_RETRIES - 1:
                    continue
                else:
                    return None

            # HTTP 200 时先解析 JSON。
            response_json = resp.json()
            print(f"[llm_client] DEBUG 响应正文片段: {json.dumps(response_json, ensure_ascii=False)[:2000]}")

            # 按约定路径提取 content。
            try:
                content = response_json["output"]["choices"][0]["message"]["content"]
                print(f"[llm_client] DEBUG 提取的 content: {content[:500]}...")
            except (KeyError, IndexError) as ex:
                print(f"[llm_client] ERROR 响应结构异常: {ex}")
                if attempt < JSON_PARSE_RETRIES - 1:
                    continue
                else:
                    return None

            # 从 content 中提取 JSON。
            result = _extract_json(content)
            if result is not None:
                print("[llm_client] DEBUG JSON 解析成功")
                return result
            else:
                print("[llm_client] DEBUG JSON 解析失败，准备重试")
                if attempt < JSON_PARSE_RETRIES - 1:
                    continue
                else:
                    return None

        except requests.RequestException as err:
            # 网络超时、DNS、连接中断等异常。
            print(f"[llm_client] ERROR 请求异常: {err}")
            traceback.print_exc()
            if attempt == JSON_PARSE_RETRIES - 1:
                return None
            print("[llm_client] DEBUG 发生异常，准备重试")
            continue

    # 下面这段是保留的兜底逻辑（在当前 return 路径下通常不会执行到）。
    # 保持原样是为了不改变原有行为。
    if response_text is None:
        print("[llm_client] ERROR response_text 为 None，放弃解析")
        return None

    # 兜底：尝试多次从 response_text 提取 JSON。
    for attempt in range(JSON_PARSE_RETRIES):
        print(f"[llm_client] DEBUG JSON 解析尝试 {attempt+1}/{JSON_PARSE_RETRIES}")
        result = _extract_json(response_text)
        if result is not None:
            print("[llm_client] DEBUG JSON 解析成功")
            return result

        # 如果包含 OpenAI 风格 choices 结构，尝试再抽一层 content。
        try:
            rsp_json = json.loads(response_text)
            print("[llm_client] DEBUG JSON 解析失败后尝试解析 choices 结构")
            if isinstance(rsp_json, dict) and "choices" in rsp_json:
                choices = rsp_json.get("choices")
                if choices and isinstance(choices, list):
                    first = choices[0]
                    content = None
                    if isinstance(first, dict):
                        content = first.get("message", {}).get("content") or first.get("text")
                    if content:
                        print(f"[llm_client] DEBUG 从 choices 抽取 content: {content[:500]}")
                        response_text = content
                        continue
        except json.JSONDecodeError as e:
            print(f"[llm_client] DEBUG JSONDecodeError: {e}")

        if attempt == JSON_PARSE_RETRIES - 1:
            print("[llm_client] ERROR 连续解析失败，退出")
            return None

    # 兜底分支：如果 result 是完整响应结构，再从 output.choices.message.content 抽问卷 JSON。
    if result is not None:
        content = result.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content")
        if content:
            print(f"[llm_client] DEBUG 提取 content: {content[:500]}")
            survey_dict = _extract_json(content)
            if survey_dict:
                print("[llm_client] DEBUG 问卷 JSON 解析成功")
                return survey_dict
            else:
                print("[llm_client] ERROR content 解析失败")
        else:
            print("[llm_client] ERROR 响应中无 content")

    return None


def call_llm_to_generate_survey(user_input: str, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    对外暴露：根据自然语言需求生成问卷。
    """
    prompt = _format_prompt_for_generation(user_input)
    survey = _call_api(prompt, model=model)
    if not survey or not isinstance(survey, dict):
        return None

    # 最基础结构校验：至少要有 title 和 questions。
    if "title" not in survey or "questions" not in survey:
        return None

    return survey


def call_llm_to_modify_survey(
    current_survey_dict: Dict[str, Any],
    modification: str,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    对外暴露：基于当前问卷 + 修改指令，返回新问卷。
    """
    if not isinstance(current_survey_dict, dict):
        return None

    prompt = _format_prompt_for_modification(current_survey_dict, modification)
    updated = _call_api(prompt, model=model)
    if not updated or not isinstance(updated, dict):
        return None

    # 基础结构校验。
    if "title" not in updated or "questions" not in updated:
        return None

    return updated
