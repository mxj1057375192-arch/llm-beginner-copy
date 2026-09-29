# 任务五 · 工具调用 Agent —— 手把手读懂版（小白向）

> 这份文档是给第一次做这个任务的你写的：**先讲清每一块在干什么、为什么这么干，再给可复制的命令**，
> 最后手把手说 git 怎么用（第 6 节），并逐行拆解代码（第 4 节）。
>
> 实验数据与踩坑记录在 [`../REPORT.md`](../REPORT.md)，几条完整 trace 在 [`traces.md`](traces.md)，
> 自检结果是 `eval/result.json`。

---

## 0. 先用大白话讲：这个任务在干什么

### 0.1 模型会"接话茬"，但不会算数、也不会查资料

本地那个 Qwen2.5-0.5B-Instruct 是一个**很小的语言模型**。它最擅长的事情是"顺着往下写"：

```
你：123 + 456 * 789 等于多少？
它：123 + 456 * 789 = 456831 左右吧……      ← 它是在"猜一个像样的回答"，不是真的在算
```

它有三个天生的短板：

1. **算术不可靠**——它不是计算器，数字是"编得像"而不是"算得对"；
2. **知识有截止日期**——训练完以后发生的事它不知道，冷门知识也记不住；
3. **它碰不到你的电脑**——它只能输出文字，不能真的去读文件、跑代码。

这个任务的思路就是：**既然它不会算、不会查，那给它配几个工具，让它自己决定什么时候用哪个。**

### 0.2 工具调用：模型"说"要干什么，我们的程序去"做"

关键要理解一件事：**模型本身什么都执行不了**。它只会写文字。

我们的程序要做的，是**跟它约定一套格式**，让它用文字说出来：

```
Thought: 这是算术题，用 calculator          ← 它的想法（写给人看，程序不解析）
Action: calculator                        ← 它想调用的工具名
Action Input: {"expression": "123 + 456 * 789"}   ← 它想传给工具的参数
```

程序读到这三行，就**真的去调用** `calculator` 这个函数，拿到结果 `456831`，
再把它写成一行 `Observation: 456831（整数，共 6 位）` 贴回对话里，
让模型看到"你上一步要的结果是这个"。

模型看到 Observation 后，要么继续调下一个工具，要么给出最终答案：

```
Thought: 我已经知道答案了
Final Answer: (123 + 456) * 789 = 456831，共 6 位
```

### 0.3 ReAct 循环长什么样

```
        ┌──────────────────────────────────────────────┐
        │  用户问题："计算 (123+456)*789 的结果和位数"     │
        └───────────────────────┬──────────────────────┘
                                ↓
        ┌──────────────────────────────────────────────┐
        │  ① 把「系统提示词 + 问题」喂给模型              │
        │     （系统提示词里写着：4 个工具叫什么、怎么调）  │
        └───────────────────────┬──────────────────────┘
                                ↓
        ┌──────────────────────────────────────────────┐
        │  ② 模型输出 Thought / Action / Action Input   │
        └───────────────────────┬──────────────────────┘
                                ↓
        ┌──────────────────────────────────────────────┐
        │  ③ 我们的程序解析出「要调哪个工具、参数是什么」  │
        │     → 真的去调用它                             │
        │     → 如果工具报错了，把错误信息也当成结果       │
        └───────────────────────┬──────────────────────┘
                                ↓
        ┌──────────────────────────────────────────────┐
        │  ④ 把结果写成 "Observation: ..." 贴回对话       │
        │     → 回到 ②，直到模型说 Final Answer          │
        │       或者步数用完（本实现上限 5 步）           │
        └──────────────────────────────────────────────┘
```

这套「想一步（Reasoning）→ 做一步（Acting）」的循环就叫 **ReAct**
（Re = Reasoning，Act = Acting）。任务要求**自己手写这个循环，不许用框架**，
因为框架会把这个过程藏起来，你就看不到它到底是怎么转的了。

### 0.4 最好理解的方式：亲眼看它跑一遍

不要先读代码。先跑第 1 节的命令，看第 2 节那份真实输出，**把"模型说的话"和"程序做的事"对上号**，
再回头读代码，一切都会很清楚。

### 0.5 必做项（DoD）与代码的对应关系

| DoD | 要求 | 代码在哪 |
|---|---|---|
| **M1** | 4 个工具，各带 OpenAI function calling 格式的 schema | [`src/tools/`](../src/tools/) 四个文件，每个都有 `TOOL_SCHEMA` 和 `run(args)` |
| **M2** | 手写 ReAct 循环：工具路由、步数上限、Final Answer 终止 | [`src/agent.py`](../src/agent.py) 的 `ReActAgent.run()` |
| **M3** | 工具抛异常时把错误塞回 Observation，别让循环崩掉 | `src/agent.py` 的 `_call_tool()` |
| **M4** | 10 题关键词命中率 > 60% | 实测 7/10 = 0.7（有波动，见第 8 节） |

---

## 1. 完全照着敲：从零到跑通

下面的命令都在 **`task-5-tool-agent` 目录下**执行（用 Git Bash）。

```bash
# ── 第 1 步：确认 Python 能用 ────────────────────────────────
# 这台机器用的是共享虚拟环境（前面任务建的），直接用完整路径最稳
"D:/新生任务/.venv/Scripts/python.exe" --version
# 应该输出：Python 3.10.x
# （如果你已经激活过虚拟环境，后面所有命令里的这段路径都可以简写成 python）

# ── 第 2 步：生成评测数据和检索夹具 ──────────────────────────
"D:/新生任务/.venv/Scripts/python.exe" data/download.py
# 应该看到：
#   已生成评测任务集：data\tasks.json（10 题）
#   已生成本地检索夹具：data\agent-fixtures

# ── 第 3 步：跑一道题，看 ReAct 循环怎么转 ────────────────────
"D:/新生任务/.venv/Scripts/python.exe" src/demo.py --task 3
# 会把 Thought / Action / Observation 全文打出来

# ── 第 4 步：跑完整自检（10 题，约 10–15 分钟）────────────────
"D:/新生任务/.venv/Scripts/python.exe" eval/run.py
```

第 4 步会打印三行结果，形如：

```
[通过] tools_individual: {"pass": true, "results": {...}}
[通过] multi_tool_success_rate: {"pass": true, "rate": 0.7, "n": 10, ...}
[跳过] error_recovery: {"skip": "需要学生实现 inject_error 测试钩子；可选实验"}
```

三种状态的含义：`[通过]` / `[跳过]`（可选或依赖网络，不算失败）/ `[失败]`。
结果同时写入 `eval/result.json`。

**不需要下载任何模型**：默认会用任务四已经下好的 `Qwen2.5-0.5B-Instruct`。
想换模型看第 7 节。

---

## 2. 一次完整的运行长什么样（逐段解读）

下面是 `python src/demo.py --task 3` 的真实输出（问题：*在 data/agent-fixtures 目录下找出所有 .md 文件，统计总数*）：

```
模型：D:\新生任务\llm-beginner\task-4-rag\models\Qwen2.5-0.5B-Instruct
问题：在 data/agent-fixtures 目录下找出所有 .md 文件，统计总数。
================================================================
```

↑ 模型加载完之后，我们把它要说的话和问题一起发给它。

```
───── 第 1 步 ─────
Thought: 首先使用 file_search 搜索数据目录下的所有 .md 文件
Action: file_search
Action Input: {"pattern": "*.md", "dir": "data/agent-fixtures"}
```

↑ **这是模型自己写出来的**。它决定用 `file_search` 这个工具，
`pattern` 给 `*.md`（只要 .md 文件），`dir` 给题目里的目录。

```
Observation: 在 data\agent-fixtures 下按「*.md」找到 2 个文件（按文件名或内容匹配）：
- data\agent-fixtures\README.md（141 字节，文件名命中）
  片段：这是任务五的本地文件检索测试文件。
- data\agent-fixtures\todo_note.md（97 字节，文件名命中）
  片段：# TODO 测试
共 2 个匹配文件。
```

↑ **这不是模型写的，是我们的程序真的去磁盘上找了一遍**，
然后把结果贴回对话。模型下一步就能"看到"这个结果。

```
───── 第 2 步 ─────
Thought: 使用观察结果来确定总共有多少个 .md 文件
Action: file_search
Action Input: {"pattern": "*.md", "dir": "data/agent-fixtures"}

Observation: 错误：上一步已经调用过同样的 file_search，结果还是：……
```

↑ 小模型犯迷糊了：**它又调了一次一模一样的工具**。
程序把它拦下来了（见 `agent.py` 里的"判重"逻辑），并提示它"要么换参数、要么直接给答案"。

```
───── 第 3 步 ─────
Thought: 我已经知道答案了
Final Answer: 2
================================================================
Final Answer：2
步数：3；agent 自报 success=True（自检不认这个字段，只按 Final Answer 的关键词判分）
这一题的期望关键词：[['2', '两']]
```

↑ 它给出了答案，循环结束（第 3 步是 `Final Answer`，不再调工具）。

**对照着看这三步，你就理解了整个任务**：模型负责"想"和"说"，程序负责"做"和"记"。

---

## 3. 代码地图：哪个文件管什么

| 文件 | 干什么 | 大概多少行 |
|---|---|---|
| [`src/paths.py`](../src/paths.py) | 路径常量：任务目录在哪、模型去哪找 | 36 |
| [`src/tools/calculator.py`](../src/tools/calculator.py) | 工具 1：算数学表达式 | 128 |
| [`src/tools/python_sandbox.py`](../src/tools/python_sandbox.py) | 工具 2：跑一段 Python 代码 | 134 |
| [`src/tools/file_search.py`](../src/tools/file_search.py) | 工具 3：找文件 / 读文件内容 | 180 |
| [`src/tools/wiki.py`](../src/tools/wiki.py) | 工具 4：查维基百科 | 157 |
| [`src/llm.py`](../src/llm.py) | 怎么把对话发给模型、怎么拿回它的回答 | 82 |
| [`src/agent.py`](../src/agent.py) | **主角**：system prompt、解析、主循环、判重、自检 | 430 |
| [`src/demo.py`](../src/demo.py) | 跑单题、打印全过程的小工具 | 70 |
| [`eval/run.py`](../eval/run.py) | 自检脚本（老师写的，不用改） | 131 |

**推荐阅读顺序**：`paths.py`（最简单，热身）→ `tools/calculator.py`（一个完整的工具长什么样）
→ `tools/file_search.py`（稍复杂）→ `llm.py` → `agent.py`（重头戏，慢慢看）。

---

## 4. 逐个文件读懂

### 4.1 `src/paths.py`——路径都写在一处

就干一件事：算出"任务目录在哪"，然后把常用路径都定成常量。
好处是别的文件不用各自写一堆 `../../../`。

```python
TASK_ROOT = Path(__file__).resolve().parents[1]   # 从 src/paths.py 往上两层 = 任务根目录
DATA_DIR = TASK_ROOT / "data"
TASKS_JSON = DATA_DIR / "tasks.json"              # 10 题评测集
```

文件末尾的 `resolve_model_dir()`：按「环境变量 `AGENT_MODEL` → 本任务 `models/` → 任务四的 `models/`」
的顺序找模型，找到哪个用哪个。**这是为了让你不用改代码就能换模型**。

### 4.2 `src/tools/calculator.py`——工具长什么样

**每个工具都是同一个套路，只要记住两块**：

```python
TOOL_SCHEMA = {          # ① 给模型看的"说明书"（OpenAI function calling 格式）
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "计算数学表达式，支持四则运算、幂、取模和 sqrt/log/exp 等常用函数。",
        "parameters": {                       # 参数叫什么、什么类型、哪些必填
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "要计算的表达式…"}},
            "required": ["expression"],
        },
    },
}


def run(args: dict) -> str:   # ② 真正干活的函数：收一个字典，返回一段字符串
    expression = str(args["expression"]).strip()
    tree = ast.parse(expression, mode="eval")
    return _format(_eval(tree))
```

`TOOL_SCHEMA` 本身不执行任何东西，它**只是要拼进 prompt 给模型看的一段说明书**。
`run()` 才是被 `agent.py` 调用的真身。

**为什么不用 `eval()`？** 这一节是新手最该注意的安全点：

```python
# ❌ 千万别这么写
eval(expression)
```

因为 `expression` 是**模型生成的文本**。模型完全可以输出
`__import__("os").system("del /f ...")` 这种东西，`eval` 会照跑不误。
所以这里改用 `ast` 把表达式**解析成一棵树**，逐个节点检查是不是白名单里的运算和函数：

```python
_FUNCS = {"sqrt": math.sqrt, "log": math.log, ...}     # 白名单：只有这些函数能调
_BINOPS = {ast.Add: operator.add, ast.Mult: operator.mul, ...}   # 只有这几个运算符认识
```

**输出格式为什么这么啰嗦？** `_format()` 对整数会写"共 6 位"，对小数会写"保留 6 位小数"：

```
456831（整数，共 6 位）
45.011110（保留 6 位小数；完整值 45.0111097397076）
```

这不是为了好看。评测里第 1 题要问"结果的位数"，第 8 题要"小数点后 6 位"——
小模型抄一长串数字经常抄错（实测把 `45.011110` 抄成 `45.011109`），
所以把**要用的那个短数字放在最前面**，让它更容易抄对。

### 4.3 `src/tools/python_sandbox.py`——跑代码的那个工具

计算器只能算表达式。要"写个函数测试若干输入"就得上这个工具：**执行模型给的 Python 代码**。

既然是执行模型给的代码，就必须设几道闸门（文件开头有详细警告）：

```python
_ALLOWED_MODULES = {"math", "statistics", "itertools", ...}   # 只允许 import 这些
_ALLOWED_BUILTIN_NAMES = "abs all any bool dict enumerate ... print range ...".split()
#       ↑ 白名单里【没有】open、exec、eval、input、__import__，所以读写文件、执行任意代码都被挡住
```

还有个有意思的机关——**用 `sys.settrace` 掐死循环**：

```python
class _Budget:
    def __call__(self, frame, event, arg):
        self.lines += 1
        if self.lines > _MAX_LINES:                 # 执行超过 20 万行
            raise _BudgetExceeded("疑似死循环")      # 直接抛异常中断
        return self
```

`sys.settrace` 本来是给调试器用的：Python 每执行一行代码，都会调用一次这个回调。
我们趁机数行数，太多就抛异常。**Windows 上没有 `SIGALRM`，线程又不能从外面杀掉**，
这是在不起子进程的前提下少数能真正掐断 `while True: pass` 的办法。

⚠️ 文件开头那段警告要认真读：**这只是教学级防护**。白名单挡不住
`().__class__.__bases__[0].__subclasses__()` 这类反射逃逸，也挡不住内存被吃光。
只用来跑自己产出的代码，别拿去跑真正不可信的输入。

另一处贴心的设计：模型经常只写个函数定义就交差，既不调用也不 `print`。
这时工具会回一句**"需要重试：……请把要算的东西 print 出来"**，
而不是干巴巴一句"没有输出"（见文件末尾那段）。

### 4.4 `src/tools/file_search.py`——找文件、读文件

这个工具有两个必须兼顾的用法（评测题里两种都出现了）：

| 题目问法 | 需要的能力 | 怎么实现 |
|---|---|---|
| "找出所有 .md 文件" | 按**文件名**找 | `_name_matches()`：子串匹配 + `fnmatch` 通配符 |
| "找包含 TODO 的文件" | 按**内容**搜 | `_search_content()`：正则搜正文，找不到再退化成子串 |
| "README.md 第一段写了什么" | 回带**内容片段** | `_first_paragraph()`：取首个非空行 |

只回文件路径是不够的——问"第一段写了什么"的题目，模型看不到内容就没法回答。

**路径安全**在 `_resolve_target()`：

```python
raw = Path(dir_arg)
candidates = [TASK_ROOT / raw, Path.cwd() / raw]     # 相对路径优先按任务目录解析
...
for root in _allowed_roots():
    if target == root or root in target.parents:     # 必须在允许的根目录里
        return target
raise ValueError(f"拒绝访问 {dir_arg}：超出允许的检索根目录")
```

如果不做这个检查，模型传一个 `dir="../../"` 就能读到工作区外的文件。

还有一处容错：如果模型把 `dir` 直接写成某个**文件**的路径（它在"读文件"类题目里常这样），
就跳过 `pattern`，直接把文件内容返回。

### 4.5 `src/tools/wiki.py`——查维基百科

这个工具最"工程"，因为它要跟一个不稳定的外部服务打交道。三处细节值得看：

**① 中英文用的是两套接口**，因为英文维基没开 TextExtracts 扩展：

```python
_zh_extract()   # 中文：prop=extracts&exintro&explaintext —— 直接拿到纯文本摘要
_en_extract()   # 英文：prop=extracts 会返回空串！改用 action=parse&section=0 再剥 HTML 标签
```

**② 精确标题查不到就搜索兜底**：

```python
def _lookup(lang, query):
    extract = fetch(query)
    if extract:
        return query, extract
    return _search(lang, query)      # 用 list=search 搜出真实标题再取一次
```

为什么必须有？评测里第 9 题问的是「Transformer (机器学习模型)」，
而中文维基上的实际标题叫「Transformer架构」，**精确查询会直接查无此条**。

**③ 重试 + 中英互备 + REST 兜底**：这台机器访问维基要经系统代理，实测偶发 TLS 断连，
所以 `_get()` 会重试 4 次，中文查不到就换英文查，主接口断了就换 REST 摘要接口。

### 4.6 `src/llm.py`——怎么跟模型说话

做的事很薄：把 `messages`（一串 role/content 的对话）喂给模型，把它的回答拿回来。

```python
def chat(self, messages, stop=None) -> str:
    if self.client is not None:          # 设了 OPENAI_BASE_URL 就走 HTTP（Ollama/vLLM）
        ...
    return self._local_chat(messages, list(stop or []))
```

进程内那条路（本机默认走这条）做了两件事：套用模型的 chat 模板、用贪心解码生成。

**最有价值的是停止串**，它是实测逼出来的：

```python
class _StopOnText(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        text = tokenizer.decode(input_ids[0][prompt_len:], skip_special_tokens=True)
        return any(marker in text for marker in stop)      # 一出现 "Observation:" 就停
```

原因：**模型会顺着 few-shot 的格式，把 Observation 自己编出来**。
实测见过它写完 `Action Input:` 之后接着写 `Observation: 2500`（假的！）——
不掐掉的话，agent 就自己跟自己对话，真工具一次都不会被调用。

### 4.7 `src/agent.py`——主角，慢慢看

430 行听起来吓人，其实分成五块，建议从上往下读：

**① system prompt（`_SYSTEM_TEMPLATE`）**——就是给模型看的"员工手册"：

```
你是一个会用工具解决问题的助手。每一步只输出下面两种格式之一。
格式一（需要调工具）：Thought / Action / Action Input
格式二（信息够了，给答案）：Thought / Final Answer
规则：1. Action 必须是工具之一，参数要写全…… 6. 题目里出现「计算」「多少年」时必须用 calculator 真算一遍
选工具的判断：- 算术 → calculator  - 要跑代码 → python_sandbox  …
可用工具：
{tool_lines}          ← 这里会被替换成 4 个工具的动态列表
示例一 ~ 示例五：……    ← few-shot 示例，教它各种情况该怎么写
```

`{tool_lines}` 是 `_tool_lines()` 从每个工具的 `TOOL_SCHEMA` **动态拼**出来的：
以后你改了工具的说明，prompt 会自动跟着变，不用手改两处。

**② 解析阶梯（`_parse_action` / `_load_json_like`）**——模型给的参数经常不规矩，所以层层退让：

```
严格 JSON          {"expression": "sqrt(2026)"}          ✅ 直接用
首个配平花括号块    前面带废话时截取                        ✅
ast.literal_eval   单引号 {'expression': 'sqrt(2026)'}    ✅
正则硬抠           漏了收尾引号 {"query": "Transformer…}   ✅ 勉强能用
裸值                只写了 2 + 3 * 4                      ✅ 单参数工具兜住
全都失败            回一句"格式错误，请重写"给模型           ↩ 让它重试
```

**③ 主循环（`run()`）**——每一轮做这几件事，按顺序看 `for index in range(self.max_steps)` 那一大段：

1. 把对话发给模型，拿到它这一步的原话 `raw`；
2. 从 `raw` 里解析出 `Thought` / `Action` / `Action Input` / `Final Answer`；
3. 有 `Action` 就去调工具（`_call_tool`），把结果当 Observation；
4. 没有 `Action` 但有 `Final Answer`，就先过一道**收尾自检**（`_finish_blocker`）：
   - 一次工具都没调成 → 打回，提示"别凭印象作答"；
   - 题目要算数字（含"计算""多少年""位数"等词）却没成功调用过计算工具 → 打回，提示"先用工具算一遍"。
5. 把 `raw` 和 `Observation: ...` 追加进对话，进入下一轮；
6. 步数用完就打住，返回 `{"steps": [...], "final_answer": ..., "success": ...}`。

**④ 判重**——如果模型连续调用**同样的工具 + 同样的参数**（且上一次是成功的），
就拦下来告诉它别重复了；但**上一次失败时允许重试**：

```python
if signature == last_signature and not last_failed:
    observation = "错误：上一步已经调用过同样的 ..."
```

这条是看 trace 时抓出来的 bug：早期版本把"失败后的原参数重试"也拦掉了，
而重试恰恰是错误恢复里最该鼓励的行为（见 [`traces.md`](traces.md) 的错误恢复一节）。

**⑤ 错误处理（`_call_tool`）**——M3 的核心就这几行：

```python
try:
    output = str(module.run(dict(action_input)))
except KeyError as exc:
    # 参数写漏是最常见的错法，直接把"该怎么补"写进 Observation
    return f"错误：{action} 缺少必需参数 {missing}。请按 'Action Input: {...}' 的形式把参数补全后重试。"
except Exception as exc:
    return f"工具 {action} 执行失败：{type(exc).__name__}: {exc}。请改参数重试或换一个工具。"
```

**任何**工具异常都被转成一段文字塞回 Observation，循环继续跑——
绝不会因为一次工具失败就让整个 `run` 崩掉。这就是 M3 要的东西。

---

## 5. 自检脚本怎么判分

`eval/run.py`（老师写的，不用改）跑三个测试：

| 测试 | 干什么 | 通过标准 |
|---|---|---|
| `tools_individual` | 单独调每个工具一次，看返回 | 四个工具都对；wiki 依赖网络，连不上算"跳过" |
| `multi_tool_success_rate` | 让 agent 跑完 10 题，逐题检查答案 | 命中率 **> 0.6** |
| `error_recovery` | 注入一次假故障，看 agent 还能不能做完 | 加分项，默认跳过 |

**判分规则**（`answer_matches()`）值得单独理解，因为它决定了很多细节：

```python
text = str(answer).lower()
text = text.replace(",", "").replace("，", "")     # 去掉逗号和全角逗号
text = re.sub(r"\s+", "", text)                    # 去掉所有空白
# 然后要求 expected_answer_contains 里的每一项都出现在答案里
# 其中 [["78", "79"]] 这种嵌套列表表示"任一即可"
```

所以：

- 第 1 题要求答案里**同时**出现 `456831` **和**（`6位`/`6 位`/`六位`/`6 digits` 之一）——
  只说"位数是 6"不行，因为它没写结果本身；
- 第 6 题要求字面 `level`、`True`、`world`、`False` 四个词——
  回答"level 是回文，world 不是回文"**不算过**（没有 True/False），这是个很挑剔但合理的检查；
- 第 8 题要 `45.011110` 或 `45.01111`，原样打印 `45.0111097397076` 一个都不命中。

**为什么"不信任 agent 自报的 success"？** 因为 `success` 是 agent 自己填的字段，
它说自己成功了不代表答案对（实测它经常理直气壮地给出编造的答案）。
判分只看 `final_answer` 里的关键词——这也是这个自检设计里最值得学的一点：
**评测要盯着可验证的结果，而不是被执行者自己的汇报。**

---

## 6. 把成果存进 git（第一次用 git 的话，看这一节）

### 6.1 git 是什么

把 git 想成**给文件夹拍快照的相机**：每拍一张（一次 commit），
以后任何时候都能翻回那一张，看清楚这次改了什么。

### 6.2 `git status` 是什么

它的意思是"**现在有哪些改动还没存进 git**"。在项目根目录敲：

```bash
git status
```

你会看到几组文件，可能带这些标记：

| 标记 | 含义 | 大白话 |
|---|---|---|
| `?? 文件名` | **未跟踪**（untracked） | "这个文件 git 还不认识，从没存过" |
| `M  文件名`（红色） | 已修改但**没放进待提交清单** | "改了，但还没选中要存" |
| `M  文件名`（绿色） | 已修改且**在待提交清单里** | "已经选中，等下一张快照" |

拿这次的结果举例，你会看到：

```
?? task-5-tool-agent/REPORT.md      ← 我新写的报告
?? task-5-tool-agent/docs/          ← 新增的文档目录（traces.md、walkthrough.md）
?? task-5-tool-agent/src/           ← 我写的全部代码
```

注意 `eval/result.json`（自检结果）**不会出现**，因为任务自带的
[`.gitignore`](../.gitignore) 里写了它的名字——`.gitignore` 就是一张"这些文件不用进 git"的清单
（自检结果是生成物，每次跑都会变，没必要存）。

### 6.3 两步走：add 然后 commit

```bash
# 第一步：把要存的文件放进"待提交清单"（暂存，staging）
git add task-5-tool-agent/src task-5-tool-agent/docs task-5-tool-agent/REPORT.md
# 想一次选中所有改动，也可以用：git add -A

# 第二步：真正拍快照，并写一句说明这次改了什么
git commit -m "task-5: 手写 ReAct agent + 四个工具，自检 7/10 通过"
```

`git add` 和 `git commit` 为什么要分两步？
因为 git 允许你**只挑一部分改动**存快照（比如今天只存代码、明天再存文档），
`add` 是"挑选"，`commit` 才是"拍照"。

### 6.4 存完怎么确认

```bash
git log --oneline -3      # 看最近 3 次快照
git status                # 应该显示 "nothing to commit, working tree clean"
```

### 6.5 后悔了怎么办

```bash
git reset --soft HEAD~1   # 只想撤销刚才那次 commit（文件内容不动），把改动退回"已暂存"
git restore --staged 文件名  # 只想把它从待提交清单里拿出来
git diff                  # 看当前还没存进快照的改动内容
```

⚠️ 一句话原则：**`git status` 是"看"，`git add` 是"挑"，`git commit` 是"存"**。
养成"改一阵子 → `git status` 看一眼 → 存一次"的习惯，出问题就能随时回退。

（本仓库里任务一~任务四的代码也大多还没提交，`git status` 里会看到一大片 `??`，
按上面的步骤一起 `git add -A` 存进去也可以。）

### 6.6 存到电脑 ≠ 传到 GitHub（push 与 fork）

`git add` 和 `git commit` **只动你电脑上的仓库**，跟 GitHub 一点关系都没有。
要把快照传到 GitHub，得再走一步 `git push`。

但这里有个坑，必须先说清楚：**本仓库的远程（origin）指向的是官方仓库**：

```bash
git remote -v
# origin  https://ghfast.top/https://github.com/nndl/llm-beginner.git (fetch)
# origin  https://ghfast.top/https://github.com/nndl/llm-beginner.git (push)
```

`nndl/llm-beginner` 是**课程的官方仓库，不是你的**。你往它推会因为没有权限而失败
（而且也不该往官方仓库推自己的作业）。要放到自己的 GitHub 上，正确流程是 **fork（复刻一份到自己账号）**：

```bash
# ── 第 1 步：在网页上 fork ────────────────────────────────
# 打开 https://github.com/nndl/llm-beginner
# 点右上角的 Fork 按钮 → 选你自己的账号 → 等几秒，你的账号下就有了一份同名仓库

# ── 第 2 步：把本地的 origin 改成【你自己的】仓库 ───────────
# <你的用户名> 换成你的 GitHub 用户名
git remote set-url origin https://github.com/<你的用户名>/llm-beginner.git
git remote -v          # 确认改成功了

# ── 第 3 步：先设置提交身份（只需一次，不然 commit 会报错）────
git config --global user.name  "你的名字或 GitHub 用户名"
git config --global user.email "你的邮箱"

# ── 第 4 步：拍照 + 上传 ─────────────────────────────────
git commit -m "task-5: 手写 ReAct agent + 四个工具"
git push -u origin master
```

几点说明：

- **提交身份**（`user.name` / `user.email`）只是写在快照上"这是谁改的"，随便填一个能认出你的就行；
- 第一次 `git push` 会要求登录 GitHub（浏览器弹窗授权，或者粘贴一个 Personal Access Token）；
- 原来的 origin 地址里带 `ghfast.top`，那是**国内加速下载**用的前缀，适合 `git pull` 拉代码，
  push 到自己的仓库时用标准的 `https://github.com/...` 地址即可；
- 如果你不打算马上传 GitHub，**只做 `add` + `commit` 也完全够**：本地有快照，随时能回退。
  作业要求的"你的 fork 仓库链接"才需要 push。

---

## 7. 自己动手改（最有效的学习方式）

```bash
# ① 换一个问题，看它怎么应对
python src/demo.py "把 data/agent-fixtures 下所有文件的名字列出来"
python src/demo.py "查维基百科 爱因斯坦 的出生年份"

# ② 只跑某一题（会顺便显示这题的期望关键词）
python src/demo.py --task 8

# ③ 换一个更大的模型（本机没 GPU，1.5B 会更慢但更聪明）
AGENT_MODEL="D:/新生任务/llm-beginner/task-5-tool-agent/models/Qwen2.5-1.5B-Instruct" \
  python src/demo.py --task 6

# ④ 改用 Ollama / vLLM 起的 OpenAI 兼容服务（README 推荐的 7B 路线）
OPENAI_BASE_URL="http://localhost:11434/v1" OPENAI_API_KEY="ollama" \
AGENT_MODEL_NAME="qwen2.5:7b-instruct" python eval/run.py

# ⑤ 想看每一步的日志（自检时的简版）
AGENT_VERBOSE=1 python eval/run.py
```

**加第 5 个工具的步骤**（比如"查天气"）：

1. 在 `src/tools/` 下新建 `weather.py`，照抄 `calculator.py` 的结构，写好 `TOOL_SCHEMA` 和 `run(args)`；
2. 在 `src/tools/__init__.py` 里加 `weather`（导入 + `__all__`）；
3. 在 `src/agent.py` 顶部的 `from .tools import ...` 里加上它，并在
   `self.tools = {...}` 那一行把它登记进去；
4. 重跑 `python eval/run.py` 看有没有把别的题搞坏。

**改 prompt**：直接改 `src/agent.py` 里的 `_SYSTEM_TEMPLATE`（规则和示例都在里面）。
这是影响成功率最大的地方——实测加一条"题目里出现「计算」时必须用 calculator 真算一遍"，
第 5 题就从失败变成了通过。

**改步数上限**：`ReActAgent(max_steps=5)`，默认 5 步。注意步数越大越慢（每步都要等模型生成）。

---

## 8. 常见疑问

**Q：为什么成功率会波动？同一份代码跑两遍结果不一样。**
A：这是本任务最真实的一面。模型是 0.5B，能力刚好卡在"能学会格式、但经常写错参数"的边缘，
同一道题这轮对下轮错很正常（实测第 6、7、9 题尤其不稳定，第 5、8 题比较稳）。
迭代过程中总分在 5/10 ~ 7/10 之间浮动。**这也是为什么任务要求"看 trace 逐题定位"**，
而不是只盯着一个总数字。

**Q：它为什么会自己编 Observation / 编答案？**
A：小模型的通病。它在"续写"而不是"执行"，所以会顺着格式编出看起来很合理的假结果
（实测编过一整段不存在的英文 README 内容）。我们的对策是：停止串 + 收尾自检 +
错误信息写得足够具体。但**它仍然会编**，这也是判分只看关键词、不看它自报 success 的原因。

**Q：报错 `ModuleNotFoundError: No module named 'src'`？**
A：命令要在 `task-5-tool-agent` 目录下执行（不是仓库根目录）。
脚本里已经做了 `sys.path` 处理，但 `cd` 错了还是会找不到。

**Q：中文输出乱码？**
A：Windows 控制台默认 GBK。自检脚本已内置处理；自己写临时脚本时加
`PYTHONIOENCODING=utf-8`，或者在脚本开头写 `sys.stdout.reconfigure(encoding="utf-8")`。

**Q：维基查不到 / 报"维基百科访问失败"？**
A：这台机器访问维基要经系统代理（`127.0.0.1:7890`），代理没开或网络抖动时会失败。
工具已经内置了 4 次重试 + 中英互备，仍失败时 agent 会把错误当 Observation 收到，
通常会改用记忆里的知识作答（所以偶尔会出现"工具没通但答案对"的情况）。
离线时 `tools_individual` 里的 wiki 会按"跳过"处理，不影响其余三项判定。

**Q：跑一轮太慢了，怎么快点？**
A：① 用 `python src/demo.py --task N` 只跑一题；② 改小 `max_steps`；
③ 换更小的模型（`AGENT_MODEL`）。全量 10 题一轮 10–15 分钟，主要在等 CPU 生成。

---

## 9. 术语小抄

| 词 | 大白话 |
|---|---|
| **token** | 模型眼里的"字块"，一个汉字约 1 个 token，一个英文单词约 1–2 个 |
| **prompt** | 发给模型的那段文字（系统提示词 + 问题 + 历史对话） |
| **few-shot 示例** | 在 prompt 里塞几个"输入→输出"的范例，教小模型照着格式写 |
| **system prompt** | prompt 里最前面那段"角色设定 + 规则"，告诉模型该怎么做事 |
| **停止串（stop）** | 生成到某个词就停下，防止模型越写越偏（这里用的是 `Observation:`） |
| **贪心解码** | 每步都选概率最高的下一个 token，结果可复现（对比"随机采样"） |
| **`max_new_tokens`** | 一次最多生成多少 token，超了就被硬截断（截断了模型自己不知道！） |
| **Observation** | 工具返回的结果，被贴回对话让模型看到 |
| **trace** | 一次完整运行的全过程记录（每一步的 Thought/Action/Observation） |
| **tool schema** | 工具的"说明书"，告诉模型这个工具叫什么、要什么参数 |
| **命中率** | 10 题里答案关键词对上了几题，本任务是 7/10 |
