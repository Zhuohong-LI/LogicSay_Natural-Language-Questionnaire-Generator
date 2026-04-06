"""
app.py 是这个项目的后端入口文件。

它主要做三件事：
1. 启动 Flask Web 服务。
2. 提供“生成问卷 / 修改问卷”的 API 路由。
3. 维护一个内存中的会话字典（session_id -> 当前问卷数据）。
"""

# uuid: 用来生成全局唯一 ID，这里用于生成 session_id。
import uuid
import os

# Dict: 类型标注（Type Hint）里常用的字典类型。
from typing import Dict

# Flask 相关对象：
# Flask: 创建应用
# jsonify: 把 Python 对象转成 JSON 响应
# render_template: 渲染 HTML 模板
# request: 读取客户端请求数据
from flask import Flask, jsonify, render_template, request

# 这两个函数来自你自己的 llm_client.py，用于调用大模型：
# 1) 根据自然语言生成问卷
# 2) 在已有问卷基础上按要求修改
from llm_client import call_llm_to_generate_survey, call_llm_to_modify_survey

# 创建 Flask 应用实例。__name__ 能让 Flask 知道当前模块的位置。
app = Flask(__name__)

# 内存级“会话存储”：
# 键（key）是 session_id（字符串），值（value）是问卷字典。
# 注意：这是临时存储，进程重启后会清空，不适合生产环境长期保存。
sessions: Dict[str, Dict] = {}


def generate_preview_text(survey_dict: Dict) -> str:
    """
    把问卷字典转换成“可读性更高”的纯文本预览字符串，方便前端直接展示。

    参数:
    - survey_dict: 问卷结构（通常来自大模型返回）

    返回:
    - str: 多行文本，包含标题、说明、题目、选项、依赖关系等
    """
    # 读取标题；如果没有 title 字段，就用默认文案。
    title = survey_dict.get("title", "(未命名问卷)")
    # 读取说明；没有 description 时给空字符串。
    description = survey_dict.get("description", "")

    # lines 用来按行拼装最终文本，最后会用 "\n" 连接。
    lines = [f"问卷标题：{title}"]
    # 只有在 description 非空时才追加“说明”这一行。
    if description:
        lines.append(f"说明：{description}")

    # 空行 + 小节标题，让输出更易读。
    lines.append("")
    lines.append("题目列表：")

    # 读取题目数组；如果 questions 不存在或为空，退化成 []，避免 for 循环报错。
    questions = survey_dict.get("questions") or []
    for q in questions:
        # 每道题都尽量“安全读取”，即使字段缺失也不崩溃。
        qid = q.get("id", "")
        qtitle = q.get("title", "")
        qtype = q.get("type", "text")
        # required 为 True -> 必答，否则可选。
        required = "(必答)" if q.get("required", False) else "(可选)"

        # 先组装题目基础信息。
        line = f"{qid}. {qtitle} [{qtype}] {required}"

        # 处理题目依赖（跳题逻辑），例如“Q3 只在 Q1 选了 A 时出现”。
        depends_on = q.get("depends_on")
        if depends_on and isinstance(depends_on, dict):
            dep_q = depends_on.get("question")
            dep_opt = depends_on.get("option")
            # 当依赖字段完整时，拼接标准依赖说明。
            if dep_q and dep_opt:
                line += f"（依赖：{dep_q}={dep_opt}）"
            else:
                # 字段不完整时，明确提示依赖格式有问题，便于排查数据结构。
                line += f"（依赖格式错误：{depends_on}）"

        # 把本题文本加入预览。
        lines.append(line)

        # 单选/多选题有 options，需要继续打印选项。
        if qtype in ("single", "multiple"):
            # enumerate(..., start=1) 让选项编号从 1 开始，更贴近用户习惯。
            for idx, opt in enumerate(q.get("options", []), start=1):
                lines.append(f"    {idx}. {opt}")

    # 把所有行用换行符拼成一个最终字符串返回。
    return "\n".join(lines)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """
    生成问卷接口（POST /api/generate）

    预期请求 JSON:
    - user_input: 用户的自然语言需求（必填）
    - model:      指定模型（可选）
    - session_id: 会话 ID（可选；不传则后端自动生成）
    """
    # 读取 JSON 请求体：
    # force=True: 即使请求头不标准，也尽量按 JSON 解析
    # silent=True: 解析失败时不抛异常，返回 None
    data = request.get_json(force=True, silent=True)

    # 参数校验：至少要有 user_input。
    if not data or "user_input" not in data:
        return jsonify({"error": "missing user_input"}), 400

    # 提取输入参数。
    user_input = data["user_input"]
    model = data.get("model")

    # 调用大模型生成问卷结构。
    survey = call_llm_to_generate_survey(user_input, model=model)
    if survey is None:
        # 这里返回 502（Bad Gateway）表示上游（LLM）处理失败。
        return jsonify({"error": "生成问卷失败"}), 502

    # session_id 复用优先：前端传了就用前端的，没传就新建一个 UUID。
    session_id = data.get("session_id") or str(uuid.uuid4())
    # 把本次问卷保存到内存会话，供后续 /api/modify 使用。
    sessions[session_id] = survey

    # 同步生成纯文本预览，前端可直接展示。
    preview = generate_preview_text(survey)
    # 返回 JSON：会话 ID + 结构化问卷 + 文本预览。
    return jsonify({"session_id": session_id, "survey": survey, "preview": preview})


@app.route("/api/template/load", methods=["POST"])
def api_template_load():
    """
    模板直载接口（POST /api/template/load）

    预期请求 JSON:
    - survey:     模板中的完整问卷结构（必填）
    - session_id: 会话 ID（可选；不传则后端自动生成）
    """
    data = request.get_json(force=True, silent=True)

    if not data or "survey" not in data:
        return jsonify({"error": "missing survey"}), 400

    survey = data.get("survey")
    if not isinstance(survey, dict):
        return jsonify({"error": "invalid survey"}), 400

    session_id = data.get("session_id") or str(uuid.uuid4())
    sessions[session_id] = survey

    preview = generate_preview_text(survey)
    return jsonify({"session_id": session_id, "survey": survey, "preview": preview})


@app.route("/api/modify", methods=["POST"])
def api_modify():
    """
    修改问卷接口（POST /api/modify）

    预期请求 JSON:
    - session_id:   要修改的会话 ID（必填）
    - modification: 修改指令（必填）
    - model:        指定模型（可选）
    """
    # 读取 JSON 请求体，行为与 /api/generate 一致。
    data = request.get_json(force=True, silent=True)

    # 参数校验：session_id 和 modification 都是必填。
    if not data or "session_id" not in data or "modification" not in data:
        return jsonify({"error": "missing session_id or modification"}), 400

    # 提取参数。
    session_id = data["session_id"]
    modification = data["modification"]
    model = data.get("model")

    # 先从内存中取出当前问卷版本。
    current_survey = sessions.get(session_id)
    if current_survey is None:
        # 找不到会话时返回 404。
        return jsonify({"error": "session_id not found"}), 404

    # 调用大模型，基于 current_survey + modification 产出新版本问卷。
    updated = call_llm_to_modify_survey(current_survey, modification, model=model)
    if updated is None:
        return jsonify({"error": "修改问卷失败"}), 502

    # 更新内存中的会话数据。
    sessions[session_id] = updated

    # 返回更新后的问卷及其文本预览。
    preview = generate_preview_text(updated)
    return jsonify({"session_id": session_id, "survey": updated, "preview": preview})


@app.route("/")
def index():
    """
    首页路由：渲染 templates/index.html。
    """
    return render_template("index.html")


# 只有直接运行 `python app.py` 时才会进入这里。
# 如果 app.py 是被其他文件 import，就不会执行下面这段。
if __name__ == "__main__":
    # host="0.0.0.0": 允许局域网访问
    # port=5000: 监听端口
    # debug=True: 开发模式（代码改动自动重载、错误页更详细）
    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
