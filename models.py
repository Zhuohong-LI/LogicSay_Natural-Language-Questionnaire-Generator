"""
models.py 用来定义“问卷数据结构”。

你可以把它理解成：
1. Question: 单个题目的数据模板
2. Survey: 整份问卷的数据模板

这样做的好处是：
- 结构更清晰（字段固定、含义明确）
- IDE 补全和类型检查更友好
- 转 JSON 时更容易统一格式
"""

# 让类型标注可以引用“尚未定义”的类名（前向引用）。
from __future__ import annotations

# dataclass: 自动帮你生成 __init__、__repr__ 等方法，少写很多样板代码。
# field: 用于给 dataclass 字段配置默认工厂等高级行为。
from dataclasses import dataclass, field

# 类型标注工具：
# Any: 任意类型
# Dict: 字典
# List: 列表
# Optional[T]: 可以是 T，也可以是 None
from typing import Any, Dict, List, Optional


@dataclass
class Question:
    """
    表示一条题目。

    字段说明：
    - id: 题号或唯一标识（例如 "q1"）
    - title: 题干文本
    - type: 题型，约定为 "single" | "multiple" | "text"
    - options: 选项列表（文本题一般为空）
    - required: 是否必答
    - depends_on: 跳题条件，格式通常是 {"question": "q1", "option": "是"}，无依赖时为 None
    """

    # 题目 ID，例如 "q1"
    id: str
    # 题目标题（题干）
    title: str
    # 题型，注释里标出约定值
    type: str  # "single", "multiple", "text"
    # default_factory=list 很重要：避免多个实例共享同一个默认列表对象。
    options: List[str] = field(default_factory=list)
    # 是否必答，默认 False（可选）
    required: bool = False
    # 依赖条件（跳转逻辑），默认没有
    depends_on: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        把 Question 对象转换成普通 dict，便于：
        - JSON 序列化
        - 作为 API 返回数据
        """
        data = {
            "id": self.id,
            "title": self.title,
            "type": self.type,
            "options": self.options,
            "required": self.required,
        }

        # 只有在存在依赖条件时才输出 depends_on。
        # 这样可以避免无意义的 "depends_on": null 出现在每题里。
        if self.depends_on is not None:
            data["depends_on"] = self.depends_on
        return data


@dataclass
class Survey:
    """
    表示整份问卷。

    字段说明：
    - title: 问卷标题
    - description: 问卷说明
    - questions: 题目列表（每个元素是 Question 对象）
    """

    # 问卷标题（必填）
    title: str
    # 问卷说明（可选，默认空字符串）
    description: str = ""
    # default_factory=list 同样用于避免共享可变默认值。
    questions: List[Question] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        把 Survey 对象转成 dict。
        其中 questions 会逐个调用 Question.to_dict()。
        """
        return {
            "title": self.title,
            "description": self.description,
            "questions": [q.to_dict() for q in self.questions],
        }
