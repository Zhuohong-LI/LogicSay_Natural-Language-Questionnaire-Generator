"""
DeepSeek official API client for the survey generator.

This module intentionally uses the OpenAI-compatible Chat Completions format
documented by DeepSeek:
https://platform.deepseek.com/api-docs/
"""

import json
import os
import re
import traceback
from typing import Any, Dict, Generator, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_MODEL_NAME = "deepseek-chat"

MODEL_ALIAS_MAP = {
    "": "deepseek-chat",
    "deepseek": "deepseek-chat",
    "deepseek-chat": "deepseek-chat",
    "chat": "deepseek-chat",
    "deepseek-reasoner": "deepseek-reasoner",
    "reasoner": "deepseek-reasoner",
    # Backward-compatible aliases from the previous UI/provider.
    "deepseek-r1": "deepseek-chat",
    "deepseek-r1-distill-llama-70b": "deepseek-chat",
    "qwen": "deepseek-chat",
    "qwen-plus-2025-07-28": "deepseek-chat",
}

TIMEOUT_SECONDS = 90
JSON_PARSE_RETRIES = 3
MAX_TOKENS = 2000
TEMPERATURE = 0.2

SYSTEM_PROMPT = (
    "你是一个专业的中文问卷设计助手。"
    "你必须只输出符合要求的 JSON，不要输出 Markdown、解释、前后缀文本或代码块。"
)

_VALID_QUESTION_TYPES = {"single", "multiple", "text"}
_TYPE_ALIASES = {
    "单选题": "single",
    "单选": "single",
    "多选题": "multiple",
    "多选": "multiple",
    "文本题": "text",
    "填空题": "text",
    "问答题": "text",
    "文本": "text",
}
_CN_NUM_MAP = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def _resolve_model_name(model: Optional[str]) -> str:
    requested = (model or "").strip()
    if requested in MODEL_ALIAS_MAP:
        return MODEL_ALIAS_MAP[requested]

    # DeepSeek official Chat Completions currently supports deepseek-chat and
    # deepseek-reasoner. Unknown frontend values are treated as legacy aliases
    # and forced to the stable default instead of being sent to the API.
    print(f"[llm_client] WARNING unsupported model '{requested}', fallback to deepseek-chat")
    return DEFAULT_MODEL_NAME


def _format_prompt_for_generation(user_input: str) -> str:
    return (
        "请根据以下用户需求生成一份问卷。\n"
        "输出必须是一个完整 JSON 对象，格式如下：\n"
        "{\"title\": str, \"description\": str, \"questions\": ["
        "{\"id\": str, \"title\": str, \"type\": \"single\"|\"multiple\"|\"text\", "
        "\"options\": [str], \"required\": bool, "
        "\"depends_on\": {\"question\": str, \"option\": str}|null}"
        "]}\n"
        "规则：\n"
        "1. 只返回 JSON 对象，不要返回 Markdown 或解释。\n"
        "2. 每道题必须有 id，按 q1、q2、q3 递增。\n"
        "3. 单选题 type 使用 single，多选题 type 使用 multiple，文本题 type 使用 text。\n"
        "4. 文本题 options 必须是空数组。\n"
        "5. depends_on 必须是 {\"question\": \"q1\", \"option\": \"选项文本\"} 或 null。\n"
        f"用户需求：{user_input}"
    )


def _cn_number_to_int(text: str) -> Optional[int]:
    if not text:
        return None
    text = text.strip()
    if text.isdigit():
        return int(text)
    if text == "十":
        return 10
    if "十" in text:
        left, right = text.split("十", 1)
        tens = _CN_NUM_MAP.get(left, 1 if left == "" else None)
        if tens is None:
            return None
        ones = _CN_NUM_MAP.get(right, 0 if right == "" else None)
        if ones is None:
            return None
        return tens * 10 + ones
    if len(text) == 1 and text in _CN_NUM_MAP:
        return _CN_NUM_MAP[text]
    return None


def _extract_question_index(text: str) -> Optional[int]:
    if not text:
        return None

    m_digit = re.search(r"第\s*(\d+)\s*题", text)
    if m_digit:
        return int(m_digit.group(1))

    m_cn = re.search(r"第\s*([零〇一二两三四五六七八九十]+)\s*题", text)
    if m_cn:
        return _cn_number_to_int(m_cn.group(1))

    m_qid = re.search(r"\bq(\d+)\b", text, flags=re.I)
    if m_qid:
        return int(m_qid.group(1))

    return None


def _extract_quoted_texts(text: str) -> List[str]:
    if not text:
        return []
    result: List[str] = []
    pattern = r"'([^']+)'|\"([^\"]+)\"|“([^”]+)”|‘([^’]+)’"
    for match in re.finditer(pattern, text):
        value = next((g for g in match.groups() if g), None)
        if value:
            result.append(value.strip())
    return result


def _split_option_items(raw: str) -> List[str]:
    raw = (raw or "").strip().strip("。；;，,")
    if not raw:
        return []

    quoted = _extract_quoted_texts(raw)
    if quoted:
        items: List[str] = []
        for q in quoted:
            items.extend(re.split(r"[、/,，,;；|]+", q))
        return [x.strip() for x in items if x.strip()]

    raw = re.sub(r"^(?:为|是|改为|改成|设为|设置为|设置成|修改为)\s*", "", raw)
    parts = re.split(r"[、/,，,;；|]+", raw)
    return [x.strip().strip("'\"“”‘’") for x in parts if x.strip()]


def _extract_target_type(text: str) -> Optional[str]:
    for key, value in _TYPE_ALIASES.items():
        if key in text:
            return value
    return None


def _extract_target_options(text: str) -> List[str]:
    m = re.search(r"选项(?:为|改为|改成|设为|设置为|设置成|修改为)?\s*[:：]?\s*(.+)", text)
    if not m:
        return []
    tail = m.group(1)
    tail = re.split(r"[。；;]", tail)[0]
    return _split_option_items(tail)


def _preprocess_modification(modification: str) -> str:
    raw = (modification or "").strip()
    if not raw:
        return ""

    text = re.sub(r"\s+", " ", raw)
    q_index = _extract_question_index(text)
    qid = f"q{q_index}" if q_index else None
    question_prefix = f"修改题目 {qid}：" if qid else ""

    if re.search(r"(删除|移除|去掉).*(第\s*[零〇一二两三四五六七八九十\d]+\s*题|题目|q\d+)", text):
        if qid:
            return f"删除题目 {qid}"
        m_qid = re.search(r"\bq(\d+)\b", text, flags=re.I)
        if m_qid:
            return f"删除题目 q{m_qid.group(1)}"

    if qid and re.search(r"(必答|必填)", text):
        if re.search(r"(取消|去掉|移除|改为可选|设为可选|非必答|可选)", text):
            return f"{question_prefix}将 required 改为 false"
        if re.search(r"(增加|加上|添加|设为|改为|改成|开启|启用|标记)", text):
            return f"{question_prefix}将 required 改为 true"

    if "选项" in text and re.search(r"(改为|改成|替换为|替换成)", text):
        quoted = _extract_quoted_texts(text)
        if len(quoted) >= 2:
            if qid:
                return f"{question_prefix}将选项 '{quoted[0]}' 修改为 '{quoted[1]}'"
            return f"将选项 '{quoted[0]}' 修改为 '{quoted[1]}'"

    if "选项" in text and re.search(r"(删除|移除|去掉)", text):
        quoted = _extract_quoted_texts(text)
        if quoted:
            if qid:
                return f"{question_prefix}删除选项 '{quoted[0]}'"
            return f"删除选项 '{quoted[0]}'"

    target_type = _extract_target_type(text)
    target_options = _extract_target_options(text)
    if qid and target_type:
        operations = [f"将 type 改为 '{target_type}'"]
        if target_options:
            operations.append(f"options 改为 {json.dumps(target_options, ensure_ascii=False)}")
        return f"{question_prefix}{'，'.join(operations)}"

    if qid and target_options:
        return f"{question_prefix}将 options 改为 {json.dumps(target_options, ensure_ascii=False)}"

    if qid and re.search(r"^(把|将)", text):
        body = re.sub(r"^(把|将)\s*(第\s*[零〇一二两三四五六七八九十\d]+\s*题|q\d+)\s*", "", text)
        body = body.lstrip("，,:： ").strip()
        if body:
            return f"{question_prefix}{body}"

    return text


def summarize_modification(modification: str) -> str:
    normalized = _preprocess_modification(modification)
    return normalized or (modification or "").strip()


def _format_prompt_for_modification(current_survey_dict: Dict[str, Any], modification: str) -> str:
    current = json.dumps(current_survey_dict, ensure_ascii=False)
    return (
        "你是一个问卷 JSON 编辑助手。给定当前问卷 JSON 和修改指令，请输出修改后的完整问卷 JSON。\n"
        "输出必须是一个完整 JSON 对象，格式如下：\n"
        "{\"title\": str, \"description\": str, \"questions\": ["
        "{\"id\": str, \"title\": str, \"type\": \"single\"|\"multiple\"|\"text\", "
        "\"options\": [str], \"required\": bool, "
        "\"depends_on\": {\"question\": str, \"option\": str}|null}"
        "]}\n"
        "规则：\n"
        "1. 只修改用户明确提到的题目或字段，未提及内容必须保持不变。\n"
        "2. 禁止重排题目顺序，除非用户明确要求调整顺序。\n"
        "3. 禁止修改未提及的题目标题、题型、选项、必答标记与依赖条件。\n"
        "4. 只返回 JSON 对象，不要返回 Markdown 或解释。\n"
        "5. depends_on 必须是 {\"question\": \"q1\", \"option\": \"选项文本\"} 或 null。\n"
        "6. 删除题目时，仅删除目标题目，其余题目原顺序保留。\n"
        f"当前问卷：{current}\n"
        f"修改指令：{modification}"
    )


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    text = text.strip()
    fallback: Optional[Dict[str, Any]] = None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            if "questions" in parsed or "title" in parsed:
                return parsed
            fallback = parsed
    except json.JSONDecodeError:
        pass

    matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    for candidate in matches:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                if "questions" in parsed or "title" in parsed:
                    return parsed
                if fallback is None:
                    fallback = parsed
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
            if isinstance(parsed, dict):
                if "questions" in parsed or "title" in parsed:
                    return parsed
                if fallback is None:
                    fallback = parsed
        except json.JSONDecodeError:
            continue

    return fallback


def _extract_json_loose(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.I)
    cleaned = cleaned.replace("```", "").strip()
    start = cleaned.find("{")
    if start >= 0:
        cleaned = cleaned[start:]
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)

    for candidate in (cleaned, cleaned.replace("'", "\"")):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def _normalize_question(question: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    if not isinstance(question, dict):
        return None

    normalized = dict(question)
    normalized["id"] = str(normalized.get("id") or f"q{index}")
    normalized["title"] = str(normalized.get("title") or "")

    qtype = str(normalized.get("type") or "text")
    if qtype not in _VALID_QUESTION_TYPES:
        qtype = "text"
    normalized["type"] = qtype

    options = normalized.get("options")
    if not isinstance(options, list):
        options = []
    normalized["options"] = [str(x) for x in options]
    if qtype == "text":
        normalized["options"] = []

    normalized["required"] = bool(normalized.get("required", False))

    depends_on = normalized.get("depends_on")
    if isinstance(depends_on, dict):
        dep_q = depends_on.get("question")
        dep_o = depends_on.get("option")
        if dep_q and dep_o:
            normalized["depends_on"] = {"question": str(dep_q), "option": str(dep_o)}
        else:
            normalized["depends_on"] = None
    else:
        normalized["depends_on"] = None

    return normalized


def _repair_survey_json(candidate: Any, current_survey_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    parsed: Optional[Dict[str, Any]] = None
    if isinstance(candidate, dict):
        parsed = candidate
    elif isinstance(candidate, str):
        parsed = _extract_json(candidate) or _extract_json_loose(candidate)

    if not isinstance(parsed, dict):
        return None

    repaired = dict(parsed)

    if not isinstance(repaired.get("title"), str) or not repaired.get("title", "").strip():
        fallback_title = current_survey_dict.get("title")
        repaired["title"] = fallback_title if isinstance(fallback_title, str) and fallback_title.strip() else "未命名问卷"

    if "description" not in repaired or repaired.get("description") is None:
        repaired["description"] = current_survey_dict.get("description", "")
    if not isinstance(repaired.get("description"), str):
        repaired["description"] = str(repaired.get("description"))

    questions_raw = repaired.get("questions")
    if not isinstance(questions_raw, list):
        questions_raw = current_survey_dict.get("questions", [])
        if not isinstance(questions_raw, list):
            questions_raw = []

    normalized_questions: List[Dict[str, Any]] = []
    for idx, q in enumerate(questions_raw, start=1):
        qn = _normalize_question(q, idx)
        if qn is not None:
            normalized_questions.append(qn)
    repaired["questions"] = normalized_questions

    if "title" not in repaired or "questions" not in repaired:
        return None
    return repaired


def _extract_content_from_response(response_json: Dict[str, Any]) -> Optional[str]:
    try:
        content = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None

    if isinstance(content, str):
        return content
    if content is None:
        return None
    return str(content)


def _build_headers() -> Optional[Dict[str, str]]:
    if not API_KEY:
        print("[llm_client] ERROR: DEEPSEEK_API_KEY is not configured")
        return None

    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def _build_payload(prompt: str, model: Optional[str] = None, stream: bool = False) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": _resolve_model_name(model),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    if stream:
        payload["stream"] = True
    return payload


def _call_api_for_content(prompt: str, model: Optional[str] = None, retries: int = JSON_PARSE_RETRIES) -> Optional[str]:
    headers = _build_headers()
    if headers is None:
        return None

    payload = _build_payload(prompt, model=model, stream=False)
    attempts = max(1, retries)

    for attempt in range(attempts):
        try:
            print(f"[llm_client] DEBUG request_url={API_URL}")
            print(f"[llm_client] DEBUG model={payload['model']}")
            print(f"[llm_client] DEBUG attempt={attempt + 1}/{attempts}")

            resp = requests.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)
            print(f"[llm_client] DEBUG status_code={resp.status_code}")
            print(f"[llm_client] DEBUG response_preview={resp.text[:1000]}")

            if resp.status_code != 200:
                print(f"[llm_client] ERROR DeepSeek response_body={resp.text[:2000]}")
                continue

            try:
                response_json = resp.json()
            except ValueError as ex:
                print(f"[llm_client] ERROR response is not valid JSON: {ex}")
                continue

            content = _extract_content_from_response(response_json)
            if content:
                return content

            print("[llm_client] ERROR choices[0].message.content not found")
        except requests.RequestException as err:
            print(f"[llm_client] ERROR request failed: {err}")
            traceback.print_exc()

    return None


def _extract_stream_delta(event_obj: Dict[str, Any]) -> str:
    try:
        delta = event_obj["choices"][0].get("delta", {})
    except (KeyError, IndexError, TypeError):
        return ""

    content = delta.get("content")
    if content is None:
        return ""
    return str(content)


def _call_api_stream(
    prompt: str,
    model: Optional[str] = None,
    retries: int = JSON_PARSE_RETRIES,
) -> Generator[str, None, None]:
    headers = _build_headers()
    if headers is None:
        return

    payload = _build_payload(prompt, model=model, stream=True)
    attempts = max(1, retries)

    for attempt in range(attempts):
        try:
            print(f"[llm_client] DEBUG request_url={API_URL}")
            print(f"[llm_client] DEBUG model={payload['model']}")
            print(f"[llm_client] DEBUG stream_attempt={attempt + 1}/{attempts}")

            with requests.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT_SECONDS, stream=True) as resp:
                print(f"[llm_client] DEBUG status_code={resp.status_code}")
                if resp.status_code != 200:
                    print(f"[llm_client] ERROR DeepSeek stream response_body={resp.text[:2000]}")
                    continue

                for raw_line in resp.iter_lines(decode_unicode=True):
                    if raw_line is None:
                        continue
                    line = raw_line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith(("event:", "id:", "retry:")):
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line:
                        continue
                    if line == "[DONE]":
                        return

                    try:
                        event_obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    delta = _extract_stream_delta(event_obj)
                    if delta:
                        yield delta
                return
        except requests.RequestException as err:
            print(f"[llm_client] ERROR stream request failed: {err}")
            traceback.print_exc()


def _call_api(prompt: str, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
    content = _call_api_for_content(prompt, model=model, retries=JSON_PARSE_RETRIES)
    if not content:
        return None

    parsed = _extract_json(content) or _extract_json_loose(content)
    if parsed is None:
        print("[llm_client] ERROR model returned content, but survey JSON could not be parsed")
        print(f"[llm_client] DEBUG raw_content_preview={content[:2000]}")
    return parsed


def call_llm_to_generate_survey(user_input: str, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not isinstance(user_input, str) or not user_input.strip():
        print("[llm_client] ERROR user_input is empty")
        return None

    prompt = _format_prompt_for_generation(user_input)
    survey = _call_api(prompt, model=model)
    if not isinstance(survey, dict):
        print("[llm_client] ERROR generation failed")
        return None

    if "title" not in survey or "questions" not in survey or not isinstance(survey.get("questions"), list):
        print("[llm_client] WARNING generation result is incomplete, trying repair")
        repaired = _repair_survey_json(survey, {"title": "未命名问卷", "description": "", "questions": []})
        if isinstance(repaired, dict) and "title" in repaired and isinstance(repaired.get("questions"), list):
            return repaired
        print("[llm_client] ERROR generation result repair failed")
        return None

    return survey


def call_llm_to_generate_survey_stream(
    user_input: str,
    model: Optional[str] = None,
) -> Generator[str, None, None]:
    if not isinstance(user_input, str) or not user_input.strip():
        print("[llm_client] ERROR user_input is empty")
        return

    prompt = _format_prompt_for_generation(user_input)
    yield from _call_api_stream(prompt, model=model, retries=JSON_PARSE_RETRIES)


def parse_survey_from_text(text: str) -> Optional[Dict[str, Any]]:
    candidate = _extract_json(text) or _extract_json_loose(text)
    if isinstance(candidate, dict):
        questions = candidate.get("questions")
        if "title" in candidate and isinstance(questions, list):
            return candidate

    repaired = _repair_survey_json(
        candidate if isinstance(candidate, dict) else text,
        {"title": "未命名问卷", "description": "", "questions": []},
    )
    if repaired is None:
        return None
    if "title" not in repaired or "questions" not in repaired:
        return None
    if not isinstance(repaired.get("questions"), list):
        return None
    return repaired


def call_llm_to_modify_survey(
    current_survey_dict: Dict[str, Any],
    modification: str,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(current_survey_dict, dict):
        return None

    normalized_modification = _preprocess_modification(modification)
    prompt = _format_prompt_for_modification(current_survey_dict, normalized_modification)

    parse_attempts = max(1, JSON_PARSE_RETRIES)
    last_content: Optional[str] = None
    last_candidate: Optional[Dict[str, Any]] = None

    for attempt in range(parse_attempts):
        content = _call_api_for_content(prompt, model=model, retries=1)
        if not content:
            print(f"[llm_client] WARNING modify attempt {attempt + 1}/{parse_attempts} returned no content")
            continue

        last_content = content
        updated = _extract_json(content) or _extract_json_loose(content)
        if updated is None:
            print(f"[llm_client] WARNING modify attempt {attempt + 1}/{parse_attempts} JSON parse failed")
            continue

        last_candidate = updated
        if "title" in updated and "questions" in updated and isinstance(updated.get("questions"), list):
            return updated

        repaired = _repair_survey_json(updated, current_survey_dict)
        if repaired is not None:
            print("[llm_client] INFO modify result repaired")
            return repaired

    repaired_fallback = _repair_survey_json(last_candidate or last_content, current_survey_dict)
    if repaired_fallback is not None:
        print("[llm_client] INFO modify fallback repair succeeded")
        return repaired_fallback

    return None
