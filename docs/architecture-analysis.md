# ALPHACODE 项目架构分析与改进建议

## 1. 项目概述

### 1.1 目标
ALPHACODE 是一个基于 MCTS (蒙特卡洛树搜索) 的代码探索系统，使用 Git 进行版本管理，LLM 作为智能代理。核心目标是：**让 LLM 自主决定何时使用 MCTS 探索代码优化方案**。

### 1.2 核心特性
- **自主决策**: LLM 判断何时需要 MCTS 探索
- **Git 状态管理**: 每个节点对应一个 Git commit
- **MAP-Elites 多样性**: 通过特征网格保持解的多样性
- **级联评估**: 语法 → 测试 → 质量 → 进度

---

## 2. 当前架构分析

### 2.1 目录结构

```
alphacode/
├── agent/           # Agent 模块 (新架构)
│   ├── unified.py   # 统一 Agent - LLM 自主决策
│   ├── base.py      # Agent 基类
│   ├── code.py      # 代码 Agent (未使用)
│   └── conversation.py  # 对话 Agent (未使用)
├── core/            # 核心模块
│   ├── controller.py # MCTS 控制器 (1020行 - 过大!)
│   ├── node.py      # 节点定义
│   └── tree.py      # 搜索树 + FeatureGrid + Islands
├── evaluation/      # 评估模块
│   └── evaluator.py # 级联评估器
├── llm/             # LLM 模块
│   ├── client.py    # OpenAI SDK 封装
│   ├── intent.py    # 意图检测
│   └── prompts.py   # Prompt 模板
├── mcts/            # MCTS 组件 (已实现但未使用!)
│   ├── selector.py  # UCB 选择器
│   ├── expander.py  # LLM 扩展器
│   └── evaluator.py # 评估器 + 缓存
├── search/          # 搜索组件 (旧代码，未使用)
├── state/           # 状态管理
│   ├── git_manager.py
│   └── session_manager.py
├── tools/           # 工具模块
│   ├── executor.py  # 工具执行器
│   └── mcts_tool.py # MCTS 探索工具
└── cli.py           # CLI 入口
```

### 2.2 核心流程

```
用户输入
    │
    ▼
UnifiedAgent.process()
    │
    ├── 简单问候 → 直接回复
    │
    └── 代码任务 → mcts_explore 工具
                      │
                      ▼
               MCTSController.solve()
                      │
                      ├── _init_session()  # 创建 Git 分支、根节点
                      │
                      └── 循环迭代:
                            │
                            ├── _select_node()    # UCB 选择
                            │
                            ├── _expand()         # LLM 生成动作
                            │
                            ├── _try_action()     # 执行动作
                            │
                            ├── _light_evaluate() # 轻量评估
                            │
                            ├── _backpropagate()  # 反向传播
                            │
                            └── _update_feature_grid()
                      │
                      ▼
               Solution (最佳代码 + 备选方案)
```

### 2.3 关键设计决策

| 决策 | 理由 | 状态 |
|------|------|------|
| 每节点 = Git commit | 可回溯、可恢复、可对比 | ✅ 实现 |
| 2 分支探索 | 平衡探索与利用 | ✅ 实现 |
| 级联评估 | 语法错误不跑测试，节省时间 | ✅ 实现 |
| MAP-Elites 特征网格 | 保持解多样性 | ⚠️ 部分实现 |
| Island 迁移 | 防止局部最优 | ⚠️ 框架有，未充分利用 |

---

## 3. 问题与债务

### 3.1 代码债务

| 问题 | 位置 | 影响 | 优先级 |
|------|------|------|--------|
| Controller 过大 | `core/controller.py` 1020行 | 难维护、难测试 | 🔴 高 |
| 模块冗余 | `mcts/` 未被使用 | 代码混乱 | 🟡 中 |
| 旧代码残留 | `search/` 目录 | 误导开发者 | 🟡 中 |
| Agent 分裂 | `agent/code.py`, `conversation.py` 未使用 | 代码冗余 | 🟢 低 |

### 3.2 架构问题

| 问题 | 描述 | 影响 |
|------|------|------|
| **缺少抽象** | Controller 直接包含选择、扩展、评估逻辑 | 难以替换策略 |
| **同步/异步混乱** | 使用 nest_asyncio 作为补丁 | 可能有隐藏 bug |
| **评估延迟高** | 每节点 2-3 次 LLM 调用 + pytest | 探索速度慢 |
| **缺少并行** | 串行探索节点 | 效率低 |

### 3.3 测试覆盖

```
tests/
└── test_basic.py  # 仅基础单元测试

缺失:
- MCTS 流程集成测试
- Agent 决策测试
- Git 状态管理测试
- 评估器测试
```

---

## 4. 改进方向

### 4.1 短期改进 (1-2 周)

#### P0: 重构 Controller
```
目标: 将 controller.py 拆分为独立组件

core/controller.py (≤200行) - 协调器
    │
    ├── mcts/selector.py     - 节点选择策略
    ├── mcts/expander.py     - 动作生成
    ├── mcts/evaluator.py    - 代码评估
    └── mcts/backpropagator.py - 反向传播
```

**具体任务:**
1. 将 `_select_node()`, `_calculate_ucb()` 移到 `mcts/selector.py`
2. 将 `_expand()`, `_generate_simple_actions()` 移到 `mcts/expander.py`
3. 将 `_light_evaluate()`, `_backpropagate()` 移到 `mcts/evaluator.py`
4. Controller 只保留 `solve()` 作为协调入口

#### P1: 使用已有的 mcts/ 模块
```python
# 当前: Controller 内联实现
def _expand(self, node):
    response = self.llm_client.generate_json_sync(...)

# 改进: 使用 mcts/expander.py
from alphacode.mcts.expander import LLMExpander

class MCTSController:
    def __init__(self, config):
        self.expander = LLMExpander(config, self.llm_client)
    
    def _expand(self, node):
        return self.expander.expand(node, self.search_tree.goal)
```

#### P2: 删除冗余代码
- 删除 `search/` 目录（旧代码）
- 删除或合并 `agent/code.py`, `agent/conversation.py`

### 4.2 中期改进 (2-4 周)

#### 并行评估
```python
# 当前: 串行评估
for action in actions:
    child = self._try_action(node, action)
    self._light_evaluate(child)

# 改进: 并行评估
import asyncio

async def _evaluate_parallel(self, children):
    tasks = [self._evaluate_async(child) for child in children]
    results = await asyncio.gather(*tasks)
    for child, result in zip(children, results):
        child.evaluation = result
```

#### 评估缓存
```python
# mcts/evaluator.py 已有 EvaluationCache
class EvaluationCache:
    def get(self, code_hash: str) -> EvaluationResult | None:
        return self.cache.get(code_hash)
    
    def set(self, code_hash: str, result: EvaluationResult):
        self.cache[code_hash] = result

# 使用: 在 _light_evaluate 中
code_hash = hashlib.md5(code.encode()).hexdigest()
cached = self.eval_cache.get(code_hash)
if cached:
    return cached
```

#### 自动测试生成
```python
# 新模块: tools/test_generator.py
class TestGenerator:
    def generate_tests(self, code: str, goal: str) -> str:
        """使用 LLM 生成测试用例"""
        prompt = f"""为以下代码生成测试用例:

目标: {goal}
代码:
```python
{code}
```

生成 pytest 测试，覆盖正常情况和边界情况。"""
        return self.llm_client.generate_sync(prompt)
```

### 4.3 长期改进 (1-2 月)

#### 1. 多模型集成
```python
class EnsembleEvaluator:
    """多模型评估器"""
    
    def __init__(self, models: list[str]):
        self.models = [LLMClient(model=m) for m in models]
    
    async def evaluate(self, code: str) -> float:
        # 并行调用多个模型
        scores = await asyncio.gather(*[
            m.evaluate(code) for m in self.models
        ])
        # 取平均或加权平均
        return sum(scores) / len(scores)
```

#### 2. 学习型选择器
```python
class LearnedSelector:
    """使用历史数据训练的选择器"""
    
    def __init__(self):
        self.model = self._load_model()
    
    def select(self, nodes: list[MCTSNode]) -> MCTSNode:
        features = [self._extract_features(n) for n in nodes]
        scores = self.model.predict(features)
        return nodes[np.argmax(scores)]
```

#### 3. 分布式探索
```python
# 使用 Ray 或 Celery 进行分布式 MCTS
@ray.remote
def explore_subtree(root: MCTSNode, iterations: int) -> Solution:
    controller = MCTSController(config)
    return controller.solve(root)

# 主节点分发任务
futures = [explore_subtree.remote(island_root, 100) 
           for island_root in islands]
results = ray.get(futures)
```

---

## 5. 建议优先级

### 第一优先级 (本周)
1. **修复 asyncio 问题**: 移除 nest_asyncio，改用纯异步设计
2. **清理冗余代码**: 删除 search/, agent/code.py, agent/conversation.py
3. **添加集成测试**: 至少覆盖核心 MCTS 流程

### 第二优先级 (下周)
1. **重构 Controller**: 拆分为 Selector/Expander/Evaluator
2. **使用 mcts/ 模块**: 替换内联实现
3. **评估缓存**: 使用已有的 EvaluationCache

### 第三优先级 (后续迭代)
1. 并行评估
2. 自动测试生成
3. 多模型集成

---

## 6. 重构后的目标架构

```
alphacode/
├── core/
│   ├── controller.py   # ≤200行，协调器
│   ├── node.py         # 节点定义
│   └── tree.py         # 搜索树 (移除 FeatureGrid/Islands)
│
├── mcts/
│   ├── selector.py     # UCB + Island 选择
│   ├── expander.py     # LLM 动作生成
│   ├── evaluator.py    # 级联评估 + 缓存
│   ├── backpropagator.py  # 反向传播
│   └── feature_grid.py # MAP-Elites 独立模块
│
├── agent/
│   └── unified.py      # 统一 Agent
│
├── tools/
│   ├── executor.py     # 工具执行
│   ├── mcts_tool.py    # MCTS 工具
│   └── test_generator.py  # 自动测试生成
│
├── llm/
│   ├── client.py       # OpenAI SDK
│   └── prompts.py      # Prompt 模板
│
├── state/
│   ├── git_manager.py  # Git 操作
│   └── session_manager.py
│
├── evaluation/
│   └── evaluator.py    # 级联评估
│
├── cli.py
└── config.py
```

---

## 7. 总结

### 当前优点
- ✅ 自主 MCTS 探索创新设计
- ✅ Git 状态管理清晰
- ✅ 级联评估高效
- ✅ 系统提示工程良好

### 主要问题
- 🔴 Controller 过大，职责不清
- 🟡 同步/异步设计混乱
- 🟡 模块冗余
- 🟢 测试覆盖不足

### 改进收益
| 改进 | 预期收益 |
|------|----------|
| Controller 重构 | 可维护性 ↑ 50% |
| 并行评估 | 速度 ↑ 2-3x |
| 评估缓存 | LLM 调用 ↓ 30% |
| 自动测试 | 正确性 ↑ 20% |