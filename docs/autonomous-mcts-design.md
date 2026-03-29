# ALPHACODE 自主 MCTS 探索设计

## 目标

实现一个自然的对话体验，当涉及代码编写时，由大模型自主决定是否进行 MCTS 代码探索。

## 当前架构 vs 目标架构

### 当前架构（硬编码）

```
用户输入 → Intent检测 → 
  ├─ CODE_TASK → CodeAgent → MCTS探索（强制）
  └─ 其他     → ConversationAgent → 直接回复
```

问题：
- Intent 检测是硬规则，LLM 无法自主决策
- 用户说"优化一下"无法触发 MCTS
- 对话和 MCTS 是分离的流程
- 缺少上下文感知（"用更好的方法重写"）

### 目标架构（LLM 自主）

```
用户输入 → 统一Agent → 
  ├─ 简单问题 → 直接回复
  ├─ 简单代码 → write工具 → 输出代码
  ├─ 复杂代码 → mcts_explore工具 → 探索优化
  ├─ 优化请求 → mcts_explore工具 → 继续探索
  └─ 代码问答 → read/grep工具 → 回答
```

核心变化：
1. **移除 Intent 检测** - LLM 自主判断任务类型
2. **MCTS 作为工具** - `mcts_explore(goal, code?, options?)`
3. **统一对话流程** - 单一 Agent，保持上下文
4. **自然交互** - 用户可以说"试试更好的方法"、"优化一下"

---

## 详细设计

### 1. MCTS 工具定义

```python
{
    "type": "function",
    "function": {
        "name": "mcts_explore",
        "description": """探索和优化代码的多种实现方案。

适用于：
- 复杂算法问题（排序、搜索、动态规划等）
- 需要性能优化的代码
- 不确定最佳实现方式的情况
- 用户要求"优化"、"试试其他方法"

不适用于：
- 简单的一行代码
- 配置文件、文档
- 已确定实现方式的简单任务
""",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "代码目标描述，如'实现快速排序算法'"
                },
                "code": {
                    "type": "string",
                    "description": "可选：初始代码，如果不提供则从头探索"
                },
                "iterations": {
                    "type": "integer",
                    "description": "探索迭代次数，默认10",
                    "default": 10
                },
                "focus": {
                    "type": "string",
                    "enum": ["performance", "readability", "correctness", "balanced"],
                    "description": "优化重点，默认balanced",
                    "default": "balanced"
                }
            },
            "required": ["goal"]
        }
    }
}
```

### 2. 统一 Agent 系统提示

```python
UNIFIED_SYSTEM = """你是 ALPHACODE，一个智能代码助手。

你有两类工具可用：

1. **普通工具** - read, write, edit, glob, grep, bash
   - 用于文件操作、信息查询、简单代码编写
   - 适合：读取文件、简单脚本、一次性任务

2. **MCTS 探索工具** - mcts_explore
   - 用于复杂代码问题的多方案探索和优化
   - 会自动尝试多种实现，找出最佳方案
   - 适合：算法实现、性能优化、不确定最佳方案时

决策指南：
- 简单代码需求 → 直接用 write 工具
- 复杂算法问题 → 用 mcts_explore 工具
- 用户要求"优化"或"试试其他方法" → 用 mcts_explore
- 读取文件或查询 → 用 read/grep/glob

示例对话：
用户: "写个快速排序"
你: 调用 mcts_explore(goal="实现快速排序算法")

用户: "读一下 config.py"
你: 调用 read(path="config.py")

用户: "写个 hello world"
你: 调用 write(path="hello.py", content="print('hello world')")

用户: "优化一下这个排序"
你: 调用 mcts_explore(goal="优化排序性能", code=当前代码, focus="performance")
"""
```

### 3. 工具执行流程

```python
class MCTSExploreTool:
    """MCTS 探索工具"""
    
    def execute(self, goal: str, code: str = None, 
                iterations: int = 10, focus: str = "balanced"):
        """
        执行 MCTS 代码探索。
        
        Returns:
            ToolResult 包含：
            - best_code: 最佳代码
            - alternatives: 备选方案
            - score: 评分
            - iterations: 实际迭代次数
        """
        # 1. 如果有初始代码，写入 program.py
        # 2. 运行 MCTS 探索
        # 3. 返回结果
```

### 4. 对话上下文管理

```python
class ConversationContext:
    """管理对话上下文"""
    
    def __init__(self):
        self.history: list[dict] = []
        self.current_code: str = None
        self.current_goal: str = None
        self.mcts_results: list[MCTSResult] = []
    
    def add_exchange(self, user: str, assistant: str, tools: list = None):
        """添加一轮对话"""
        
    def get_context_for_mcts(self) -> dict:
        """获取 MCTS 需要的上下文"""
        return {
            "goal": self.current_goal,
            "code": self.current_code,
            "previous_attempts": self.mcts_results,
        }
```

---

## 实现计划

### Phase 1: MCTS 工具化 (3天)

**目标**: 将 MCTS 封装为工具

**任务**:
1. 创建 `mcts_explore` 工具定义
2. 实现 `MCTSExploreTool` 类
3. 更新 `TOOL_DEFINITIONS`
4. 修改 `ToolExecutor` 支持 MCTS 工具

**文件**:
- `alphacode/tools/mcts_tool.py` (新建)
- `alphacode/tools/executor.py` (修改)

### Phase 2: 统一 Agent (3天)

**目标**: 创建统一的对话 Agent

**任务**:
1. 创建 `UnifiedAgent` 类
2. 实现 `UNIFIED_SYSTEM` 提示
3. 移除 Intent 检测逻辑
4. 实现上下文管理

**文件**:
- `alphacode/agent/unified.py` (新建)
- `alphacode/cli.py` (修改)

### Phase 3: CLI 集成 (2天)

**目标**: 更新 CLI 使用统一 Agent

**任务**:
1. 简化 `_process_input` 方法
2. 移除 `_run_mcts_exploration` 独立调用
3. 更新命令处理（/best, /alternatives 等）
4. 添加 /continue 命令继续探索

**文件**:
- `alphacode/cli.py` (修改)

### Phase 4: 测试和优化 (2天)

**目标**: 确保稳定性和用户体验

**任务**:
1. 测试各种对话场景
2. 优化 MCTS 工具响应格式
3. 改进错误处理
4. 完善日志和调试

---

## 使用示例

### 场景 1: 简单代码

```
用户: 写个 hello world
助手: [调用 write 工具]
     已创建 hello.py，内容为 print('hello world')
```

### 场景 2: 复杂算法

```
用户: 实现快速排序
助手: [调用 mcts_explore(goal="实现快速排序算法")]
     
     探索完成！找到了 3 种实现方案：
     
     最佳方案 (评分 0.92):
     ```python
     def quicksort(arr):
         if len(arr) <= 1:
             return arr
         pivot = arr[len(arr) // 2]
         left = [x for x in arr if x < pivot]
         middle = [x for x in arr if x == pivot]
         right = [x for x in arr if x > pivot]
         return quicksort(left) + middle + quicksort(right)
     ```
     
     输入 /alternatives 查看其他方案
```

### 场景 3: 优化请求

```
用户: 优化一下性能
助手: [调用 mcts_explore(goal="优化性能", code=当前代码, focus="performance")]
     
     已完成优化！新版本比原版本快 30%。
     
     优化后代码:
     ...
```

### 场景 4: 继续探索

```
用户: 还有其他方法吗？
助手: [调用 mcts_explore(goal="快速排序", previous_attempts=...)]
     
     找到了新的实现方案！
     ...
```

---

## 技术细节

### MCTS 工具输出格式

```python
@dataclass
class MCTSExploreResult:
    """MCTS 探索结果"""
    success: bool
    best_code: str
    best_score: float
    alternatives: list[dict]  # [{"code": ..., "score": ...}, ...]
    iterations: int
    message: str  # 给用户的友好消息
```

### Agent 响应格式

```python
@dataclass
class AgentResponse:
    """Agent 响应"""
    content: str  # 给用户的文本
    tool_calls: list[dict]  # 工具调用记录
    tool_results: list[dict]  # 工具结果
    mcts_used: bool = False  # 是否使用了 MCTS
    mcts_result: MCTSExploreResult = None  # MCTS 结果
```

### CLI 显示逻辑

```python
def _show_response(self, response: AgentResponse):
    """显示 Agent 响应"""
    # 1. 如果用了 MCTS，特殊显示
    if response.mcts_used:
        self._show_mcts_result(response.mcts_result)
    
    # 2. 显示普通工具结果
    for tr in response.tool_results:
        if tr["tool"] != "mcts_explore":
            print(tr["result"])
    
    # 3. 显示 Agent 文本回复
    if response.content:
        print(response.content)
```

---

## 迁移策略

### 兼容性

- 保留 `IntentDetector` 作为可选功能
- 保留 `CODE_TASK` intent 用于日志记录
- 保留 `/tree`, `/best` 等命令

### 渐进迁移

1. 先添加 `mcts_explore` 工具
2. 创建新 CLI 选项 `--unified` 使用新架构
3. 测试稳定后替换默认行为
4. 移除旧的 Intent 检测代码

---

## 风险和缓解

### 风险 1: LLM 过度使用 MCTS

**缓解**: 在系统提示中明确指导何时使用，设置工具描述中的"不适用于"场景

### 风险 2: LLM 不使用 MCTS

**缓解**: 
- 系统提示中的示例强调复杂算法用 MCTS
- 工具描述突出 MCTS 的价值
- 用户可以明确说"用 MCTS 探索"

### 风险 3: 对话上下文丢失

**缓解**:
- 保持完整的对话历史
- MCTS 结果保存到上下文
- 支持用户说"基于刚才的结果"

---

## 成功指标

1. **自然交互**: 用户可以说"优化一下"触发 MCTS
2. **智能决策**: LLM 正确判断何时需要探索
3. **上下文感知**: 可以基于之前的结果继续探索
4. **用户体验**: 减少不必要的 MCTS 启动，提高响应速度