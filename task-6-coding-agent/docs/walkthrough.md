# 任务六 · Mini Coding Agent —— 手把手读懂版（小白向）

> 这份文档是给第一次做这个任务的你写的：**先讲清每一块在干什么、为什么这么干，再给可复制的命令**，
> 然后逐个文件拆代码（第 4 节）。
>
> 实验数据与踩坑记录在 [`../REPORT.md`](../REPORT.md)，一次完整运行的 trace 在 [`traces.md`](traces.md)，
> 自检结果是 `eval/result.json`。
>
> **这份任务是全系列难度跳变最大的一次**：子系统最多，而且教材里没有对应章节兜底。
> 所以别指望一遍看懂——建议先照第 2 节跑通，再回头从第 0 节读原理。

---

## 0. 先用大白话讲：这个任务在干什么

### 0.1 目标：自己造一个"能改代码的 agent"

你可能用过 Claude Code 或者 Cursor 这类工具：你打一句话"这个函数算错了，修一下"，
它就自己去读文件、改代码、跑测试，失败了还会再改，直到跑通。

这个任务就是**用本地的小模型，把这件事的最小版本复刻一遍**。

和任务五的区别很大：

| | 任务五（工具调用 Agent） | 任务六（编程 Agent） |
|---|---|---|
| 工具干什么 | 算数、查百科、找文件——**只读** | 读文件、**改文件**、**跑测试** |
| 危险程度 | 弄错了顶多答案不对 | **会真的改坏你的代码** |
| 循环终止 | 拿到答案就停 | 得**测试全绿**才算完 |
| 上下文 | 几轮就结束 | 十几轮，还会爆 |

最后一行是关键：任务五的 agent 转两三圈就出答案了；任务六的 agent 要反复"改—测—再改"，
**对话历史会越来越长**，而模型的上下文窗口是有限的。这是新的工程约束。

### 0.2 三层栈：Tools / Skills / Subagents 各解决什么

任务书要求把能力拆成三层。别被名字唬住，用一句话记住每层的职责：

```
┌─────────────────────────────────────────────────────────┐
│  顶层：Subagents（子 agent）                             │
│  独立上下文的分身。主 agent 不看它的过程，只看它的结论。    │
│  解决：子任务过程太长，会把主 agent 的上下文撑爆            │
├─────────────────────────────────────────────────────────┤
│  中层：Skills（技能包）                                   │
│  一段"遇到这类活该按什么流程做"的说明文字，按需加载。        │
│  解决：把所有经验都写进系统提示太长，模型反而抓不住重点      │
├─────────────────────────────────────────────────────────┤
│  底层：Tools / MCP（原子工具）                            │
│  读文件、写文件、跑测试——一个个具体动作。                   │
│  解决：模型自己碰不到你的电脑，只能"说"，得有人替它"做"      │
└─────────────────────────────────────────────────────────┘
```

**打个比方**：Tools 是扳手螺丝刀，Skills 是"修水管先关总闸"这类作业规程，
Subagents 是你派出去跑腿的徒弟——他跑一趟回来只跟你说结论，不用把路上的每一步都汇报给你。

> ⚠️ 本实现**只做了下面两层**，Subagent 层没做（见 `REPORT.md` 的「未完成」一节）。
> 这是有意取舍：DoD 必做的 4 项不包含它，先把必需项跑通再加。

### 0.3 MCP 是什么：把工具做成"标准插座"

你可能会想：工具不就是几个 Python 函数吗，直接 import 调用不行吗？

行，但那样工具和 agent 就**焊死**在一起了。MCP（Model Context Protocol）是一套约定好的协议，
把工具做成**独立的服务**，agent 通过标准协议去调用它。好处是：

- 工具可以换成别人写的（只要他也遵守 MCP），agent 代码一行不用改；
- 工具可以跑在别的机器/别的语言里；
- 你可以用现成的 MCP 客户端（比如 Claude Desktop）来测你的工具对不对。

通信方式有好几种，本任务用的是最简单的 **stdio**：agent 起一个工具服务的子进程，
两边通过**标准输入输出**互相发 JSON-RPC 消息。

```
   agent 进程                          mcp_server 子进程
       │                                      │
       │  ── 启动：python src/mcp_server.py ──→ │
       │                                      │
       │  ── {"method":"tools/list"} ────────→ │
       │  ←──────── {"result":[6个工具]} ────── │
       │                                      │
       │  ── {"method":"tools/call",          │
       │      "params":{"name":"read_file",   │
       │                "arguments":{...}}} ─→ │  真的去读文件
       │  ←──────── {"result":"文件内容"} ───── │
```

⚠️ **一个必须记住的坑**：stdio 模式下 **stdout 被 JSON-RPC 独占**。
你的 server 里任何一句 `print("调试信息")` 都会污染协议、让客户端解析失败。
要打印调试信息只能写 `print(..., file=sys.stderr)`——这就是 `mcp_server.py` 里
那句 `print` 特意加 `file=sys.stderr` 的原因。

### 0.4 agent loop 长什么样

```
        ┌──────────────────────────────────────────────┐
        │  issue:"修复 calculator.add，让 pytest 全绿"   │
        └───────────────────────┬──────────────────────┘
                                ↓
        ┌──────────────────────────────────────────────┐
        │  把 issue + 工具说明 + 命中的 Skill 拼成提示词  │
        └───────────────────────┬──────────────────────┘
                                ↓
              ┌─────────────────────────────────┐
        ┌────→│  模型输出（一轮只输出一组）：      │
        │     │  Thought: 先看看 add 怎么写的     │
        │     │  Action: read_file              │
        │     │  Action Input: {"path":"..."}   │
        │     └────────────────┬────────────────┘
        │                      ↓
        │     ┌─────────────────────────────────┐
        │     │  我们的程序解析这三行             │
        │     │  → 通过 MCP 真的去调用工具        │
        │     │  → 拿到结果                       │
        │     └────────────────┬────────────────┘
        │                      ↓
        │     ┌─────────────────────────────────┐
        │     │  结果写成 Observation 贴回对话     │
        │     │  Observation: {"ok":true,...}   │
        │     └────────────────┬────────────────┘
        │                      ↓
        │              测试通过了吗？
        │              没通过 ↓        ↓ 通过了
        └──────────────────┘        停机，输出 trace
```

注意最后那个判断——**停机条件必须是程序说了算，不能听模型自己说"我修好了"**。
这一点非常关键，第 4.5 节会讲我们踩的坑。

### 0.5 DoD 必做项与代码的对应关系

| DoD | 要求 | 代码在哪 | 自检怎么验 |
|---|---|---|---|
| **M1** | MCP server，导出 `list_tools()`，≥5 个工具 | `src/mcp_server.py` | import `list_tools()` 数个数 |
| **M2** | Skill 加载器 + 2-3 个 `SKILL.md`，每个有 name+description | `src/skill_loader.py` + `src/skills/` | 扫描目录查元数据齐不齐 |
| **M3** | `CodingAgent.run()` 返回 Trace，修好 toy-repo | `src/agent.py` | 真跑一遍，再用 pytest 验证 |
| **M4** | trace 记录 thought / tool_call / observation | `src/agent.py` | 查 `trace.get("steps")` 等键 |

---

## 1. 预备：先把模型搞定（本机没有 GPU 是最大的坎）

### 1.1 三个选择，以及为什么最后选了最小的那个

任务书点名要 **Qwen2.5-Coder-7B-Instruct**。但这台机器是 i5-1135G7（4 核 8 线程）+ 16GB 内存，
**没有独立显卡**。三种方案的账是这样的：

| 方案 | 体积 | CPU 上速度 | 结论 |
|---|---|---|---|
| 7B FP16 | ~15 GB | 内存装不下 | ❌ 直接排除 |
| 7B Q4_K_M | ~4.7 GB | 约 2 tok/s | ⚠️ 能跑，但一轮对话几十秒，调不动 |
| **1.5B Q4_K_M** | **1.1 GB** | **实测 25 tok/s** | ✅ 选它 |

"tok/s" 是每秒生成多少个 token（可以粗略理解成"字"）。为什么这个数字要命：

一次 agent 循环要调用模型十来次，每次生成 200-300 个 token。按 2 tok/s 算，
**一次完整修复要跑几十分钟**，而你调试时得反复跑几十次——一天就没了。
按 25 tok/s 算，**21 秒一次**，改一版试一版完全没压力。

选 1.5B 是**工程上的正确取舍**：toy-repo 的 bug 是 `return a - b` 写成了该写 `a + b`，
这种程度的活 1.5B 完全够用。等你要挑战 SWE-bench（真仓库真 issue）时再考虑上大模型。

### 1.2 装依赖

```bash
# 注意：这个 shell 里直接敲 python 会用到 Anaconda 的 base 环境，
# 那个环境里没装 mcp。所以务必用虚拟环境的完整路径。
"D:/新生任务/.venv/Scripts/python.exe" -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  mcp openai pytest
```

> `-i` 后面是清华的 pip 镜像，国内下载快很多。已经装过的话这条命令会秒过。
>
> 任务书的 `requirements.txt` 里还有 `datasets` / `pyarrow` / `pandas`，
> 那三个只有跑 SWE-bench 加分项才用得上。自检脚本很贴心——parquet 文件不存在时
> 它直接跳过、**根本不会 import pandas**，所以不装也不影响必做项。

### 1.3 下模型 + 起服务

模型和推理引擎都放在**仓库外面**的 `D:\新生任务\.tools\`，这样不会污染你的提交：

```bash
# ① 下模型（1.1GB，ModelScope 实测 7.3MB/s，约两分半）
#    脚本见 D:\新生任务\.tools\fetch_model.py，带断点续传，断了重跑即可

# ② 起推理服务（llama.cpp 的 CPU 版）
"D:/新生任务/.tools/llama-cpp/llama-server.exe" \
  -m "D:/新生任务/llm-beginner/task-6-coding-agent/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" \
  --host 127.0.0.1 --port 8080 -c 8192 -t 4 --jinja
```

参数逐个解释：

| 参数 | 意思 |
|---|---|
| `-m <文件>` | 模型权重文件（.gguf 格式，把权重和量化信息打包在一起） |
| `--port 8080` | 监听端口，待会儿 agent 就连这里 |
| `-c 8192` | 上下文长度 8192 个 token，够放十几轮对话 |
| `-t 4` | 用 4 个线程（这台机器 4 核 8 线程，物理核数最划算） |
| `--jinja` | 用模型自带的对话模板，保证多轮对话格式正确 |

**这个窗口不要关**，它就是个服务器，得一直开着。另开一个窗口干活。

验证一下服务活了没：

```bash
curl http://127.0.0.1:8080/health
# 应该返回：{"status":"ok"}
```

---

## 2. 完全照着敲：从零到跑通

```bash
# ── 第 0 步：确认你在任务目录里 ──────────────────────────────
cd "D:/新生任务/llm-beginner/task-6-coding-agent"

# ── 第 1 步：生成 toy repo（一个专门用来练手的迷你代码仓库）────
"D:/新生任务/.venv/Scripts/python.exe" data/download.py
# 应该看到：已生成本地 toy repo：data\toy-repo
# 顺带打印一堆模型部署提示，那些是给没配好模型的人看的，忽略即可

# 看看它生成了什么
ls data/toy-repo/
# ISSUE.md              ← 任务描述："修复 calculator.add，让 pytest 全绿"
# calculator.py         ← 有 bug 的代码（add 里写成了 a - b）
# calculator.py.orig    ← bug 版本的原样快照，自检每次靠它恢复
# test_calculator.py    ← 测试，不许改

# ── 第 2 步：确认模型服务在跑（上一个窗口别关）───────────────
curl http://127.0.0.1:8080/health

# ── 第 3 步：跑一次 agent，看它怎么干活 ─────────────────────
"D:/新生任务/.venv/Scripts/python.exe" -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
import shutil
from src.agent import CodingAgent
TOY = Path('data/toy-repo').resolve()
shutil.copy(TOY/'calculator.py.orig', TOY/'calculator.py')   # 恢复成有 bug 的样子
trace = CodingAgent(verbose=True).run(str(TOY), (TOY/'ISSUE.md').read_text(encoding='utf-8'))
print('测试通过:', trace['tests_passed'])
print('停机原因:', trace['done_reason'])
print('步数:', len(trace['steps']))
"
# 应该看到三行 [step 0/1/2]，最后 测试通过: True
# 大约 20 秒

# ── 第 4 步：跑完整自检（这才是判分的那个）──────────────────
"D:/新生任务/.venv/Scripts/python.exe" eval/run.py
```

第 4 步应该输出：

```
[通过] mcp_server_lists_tools: {... "tools": ["read_file", "write_file", "edit_file", ...]}
[通过] skill_loader_metadata:  {... "count": 3, "missing_meta": []}
[通过] toy_repo_patch:         {... "tests_passed": true, "trace_has_patch": true}
[跳过] swebench_lite_sample:   {... "skip": "data/swebench-lite-sample.parquet 不存在"}
```

三态的含义：**通过** = 契约满足；**跳过** = 前置条件没就绪，**不是错误**（这里是因为没下
SWE-bench 数据）；**失败** = 实现和预期不符。结果同时写进 `eval/result.json`。

---

## 3. 代码地图：哪个文件管什么

```
task-6-coding-agent/
├── src/
│   ├── mcp_server.py      ← M1：6 个工具 + MCP 服务（独立进程，被 agent 启动）
│   ├── skill_loader.py    ← M2：扫描 SKILL.md，按 description 匹配
│   ├── skills/            ← M2：三个技能包，每个一个 SKILL.md
│   │   ├── test-runner/SKILL.md
│   │   ├── code-review/SKILL.md
│   │   └── pr-description-writer/SKILL.md
│   ├── llm.py             ← 跟模型说话的封装（发 HTTP 请求）
│   └── agent.py           ← M3+M4：主循环，主角
├── data/toy-repo/         ← 练手用的小仓库（download.py 生成，不进 git）
├── eval/run.py            ← 自检脚本（判分的就是它）
└── models/                ← 模型权重（不进 git）
```

调用关系：

```
agent.py
  ├─ 启动子进程 → mcp_server.py（工具都在这）
  ├─ 用 skill_loader.py 找相关 Skill
  └─ 用 llm.py 跟 llama-server 说话
```

---

## 4. 逐个文件读懂

### 4.1 `src/mcp_server.py`——6 个工具 + 三条安全线

**这是本任务最需要注意安全的文件**——因为 agent 会用它读写文件、执行命令，
模型一旦产生奇怪的路径，就可能读写到仓库外面去。

#### 工具清单

| 工具 | 干什么 | 为什么要它 |
|---|---|---|
| `read_file` | 读文件（带行号） | 模型得先看见代码 |
| `edit_file` | 把文件里的一段文字替换掉 | **改 bug 的主力**，见下面的说明 |
| `write_file` | 整个文件覆盖写 | 新建文件用 |
| `run_tests` | 跑 `python -m pytest -q` | 验证改对没有 |
| `git_diff` | 看工作区改了什么 | 让 agent 确认自己的改动 |
| `git_apply` | 打一个补丁 | 复杂改动用 |

> **为什么要有 `edit_file`？** 最初只有 `write_file`，结果 1.5B 模型老出错——
> 它得把整个文件内容塞进 JSON 字符串，里面的换行要转义成 `\n`，
> 模型经常在这儿翻车。改成 `edit_file` 只需要给两个小片段（原文、替换成），
> 出错概率骤降，一次就过。

#### 安全线一：路径不能跑出仓库

```python
def _resolve_in_repo(rel: str) -> Path:
    root = _repo_root()
    candidate = Path(rel)
    if candidate.is_absolute():
        raise ValueError(f"只接受相对路径，收到绝对路径：{rel}")
    target = (root / candidate).resolve()          # ← resolve 会展开 .. 和符号链接
    if target != root and root not in target.parents:
        raise ValueError(f"路径越界，拒绝访问 repo 外的文件：{rel}")
    return target
```

逐行看：

1. `Path(rel)` —— 把模型给的字符串变成路径对象；
2. `is_absolute()` —— 挡掉 `D:/windows/win.ini` 这种绝对路径；
3. `(root / candidate).resolve()` —— **关键**。`.resolve()` 会把 `..` 和符号链接**真正展开**。
   比如模型传 `../../secret.txt`，展开后就成了仓库外面的路径；
4. `root not in target.parents` —— 展开**之后**再看它是不是还在仓库里面。

⚠️ **顺序不能反**。如果先检查字符串、再 resolve，`../../` 就能绕过去。
这是路径穿越漏洞最常见的写法错误。

#### 安全线二：执行命令不用 shell

```python
proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                      timeout=timeout, shell=False)
```

三个要点：

- `shell=False` + **列表形式**传参数。如果写成 `shell=True` 加字符串拼接，
  模型给出的文件名里只要带个 `; rm -rf /` 就可能被执行——这叫 shell 注入；
- `cwd` 限定工作目录；
- `timeout` 防止命令卡死把整个 agent 拖住。

#### 安全线三：git 命令白名单

`git_apply` 是唯一会改动仓库的命令，而且先预检、失败不落盘：

```python
check = subprocess.run(["git", "apply", "--check", "-"], ...)   # 先试一下能不能打上
if check.returncode != 0:
    return _err(f"patch 预检失败，未落盘：...")                  # 打不上就什么都不做
applied = subprocess.run(["git", "apply", "-"], ...)            # 确认能打上才真打
```

同时**刻意不支持** `git reset --hard` / `git clean -fd` / `git checkout --` 这些命令
——它们会丢掉未提交的改动、删掉没纳入版本控制的文件。给 agent 的工具集要"够用就好"，
能删数据的命令一个都不要给。

#### 错误必须变成返回值，不能让服务崩

```python
def call_tool(name: str, arguments: dict) -> dict:
    spec = _BY_NAME.get(name)
    if spec is None:
        return _err(f"未知工具：{name}（可用：{', '.join(_BY_NAME)}）")
    try:
        return spec["fn"](**(arguments or {}))
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}")     # ← 异常变成返回值
```

如果工具抛异常让进程挂了，agent 那边只会看到"连接断开"，完全不知道发生了什么、
更没法自我纠正。把异常转成 `{"ok": false, "error": "..."}` 回给模型，
模型看到错误信息还能再试一次——这就是**错误恢复**。

#### 工具表是"唯一事实来源"

```python
TOOL_SPECS = [
    {"name": "read_file", "description": "...", "input_schema": {...}, "fn": op_read_file},
    ...
]

def list_tools() -> list[dict]:
    """自检入口：同步返回工具清单。"""
    return [{"name": s["name"], "description": s["description"],
             "input_schema": s["input_schema"]} for s in TOOL_SPECS]
```

工具清单只写**一份**，`list_tools()`（自检用）和 MCP 服务注册（真调用用）都从它生成。
两处各写一份的话，早晚会不一致。

### 4.2 `src/skill_loader.py`——渐进式披露

**要解决的问题**：你希望 agent 懂"测试失败该怎么排查"这套流程。最直接的办法是把流程
写进系统提示——但如果把代码审查、写 PR 描述、测试诊断全写进去，提示词就长得离谱，
小模型反而抓不住重点。

**办法**：每个技能包只用一句话的 `description` 说明"什么时候该用它"。
平时只让模型看到这些一句话；**真需要时，才把那个技能包的完整正文load 进来**。
这就是"渐进式披露"（progressive disclosure）。

```python
def list_skills(self) -> list[dict]:
    """只返回元数据（name / description / path），不读正文。"""
    found = []
    for path in sorted(self.skills_dir.glob("*/SKILL.md")):
        meta, _ = _split_front_matter(path.read_text(encoding="utf-8", errors="replace"))
        name = meta.get("name") or path.parent.name
        found.append({"name": name,
                      "description": meta.get("description") or "",
                      "path": str(path)})
        self._by_name[name] = path
    return found
```

注意 `_split_front_matter` 返回两个值，这里用 `_` 丢掉了第二个（正文）
——**读了文件但只取元数据**，正文等到 `load()` 时才用。

匹配用的是最朴素的"词重叠"：

```python
def match(self, task: str, top_k: int = 1) -> list[dict]:
    wanted = _tokens(task)
    scored = []
    for meta in self.list_skills():
        score = len(wanted & _tokens(f"{meta['name']} {meta['description']}"))
        if score > 0:
            scored.append((score, meta))
    scored.sort(key=lambda pair: -pair[0])
    return [meta for _, meta in scored[:top_k]]
```

`_tokens` 把文本拆成"英文单词 + 单个汉字"的集合，取交集算分。
没上语义模型，够用且零成本——反正 description 是你自己写的，把关键词写全就行。

### 4.3 `src/skills/*/SKILL.md`——技能包长什么样

格式是 **YAML front-matter + 正文**，和 Anthropic 的 Skills 约定一致：

```markdown
---
name: test-runner
description: 当需要运行测试、判断改动是否让测试通过、或诊断 pytest 失败原因时加载。改完代码要验证时必须先加载它。
---

# 测试运行与失败诊断

1. 先调 `run_tests` 拿真实结果，**不要凭猜**判断测试过没过。
2. 输出里全绿 → 任务完成，回报 patch 与测试结论。
3. 出现 `failed`：读完整 traceback，定位到具体断言、文件与行号，再决定改哪里。
4. **只改被测代码，不要改测试文件。**
```

上面 `---` 之间的部分就是 front-matter，两个必填字段：

- `name`：技能名，`load()` 时按它找；
- `description`：**必须写清"什么时候加载"**，而不是"这个技能是干什么的"。
  对比一下：
  - ❌ `description: 测试相关` —— 太泛，匹配不上任何具体任务
  - ✅ `description: 当需要运行测试、诊断 pytest 失败原因时加载` —— 任务里出现"跑测试"
    就能命中

### 4.4 `src/llm.py`——怎么跟模型说话

llama-server 起的是一个 **OpenAI 兼容**的 HTTP 服务，意思是它的接口格式和 OpenAI 官方一样，
所以可以直接用 `openai` 这个包：

```python
self.client = OpenAI(base_url=self.base_url, api_key=..., timeout=600)

resp = self.client.chat.completions.create(
    model=self.model_name, messages=messages,
    temperature=self.temperature, max_tokens=self.max_tokens,
    stop=list(stop) if stop else None)
```

两个参数值得注意：

- `temperature=0`：每次选概率最高的词，**输出可复现**。agent 任务要的是稳定，不要创意；
- `stop=["\nObservation"]`：让模型**生成到这儿就停**。

关于 `stop` 有个真实的教训：小模型看到提示词里有 `Observation:` 的格式，
会**自己往下编 Observation**（假装自己已经拿到工具结果了）。不掐掉的话 agent 就变成
"自问自答"，一个真工具都不会调用。所以生成到 `Observation` 就强行截断。

### 4.5 `src/agent.py`——主角，慢慢看

#### 4.5.1 为什么不用 JSON tool-calling

现在很多模型支持原生 tool-calling（把工具定义成 JSON schema 传进去，模型返回结构化的
函数调用）。但 1.5B 这个级别的模型做这件事很不稳，尤其是参数里带换行（写文件内容）时，
转义经常出错。

所以这里用的是更"土"但更稳的 **ReAct 文本协议**：让模型输出三行文字，我们自己去解析。

```
Thought: 先读一下 calculator.py             ← 它的想法（给人看，程序不解析）
Action: read_file                          ← 工具名（程序要解析）
Action Input: {"path": "calculator.py"}    ← 参数（程序要解析，是 JSON）
```

好处是**格式扁平、好解析、好纠错**——解析失败时把"格式要求"当成 Observation 喂回去，
模型下一轮往往就能改对。

#### 4.5.2 主循环骨架

```python
async def _run(self, repo_path: str, issue: str) -> dict:
    repo = Path(repo_path).resolve()
    before = _snapshot(repo)                       # 改动前拍个快照，后面算 patch
    system, trace["skills_used"] = self._build_system(issue)

    env = dict(os.environ)
    env["AGENT_REPO_ROOT"] = str(repo)             # 告诉工具服务：安全边界是这个目录
    params = StdioServerParameters(command=sys.executable,
                                   args=[str(MCP_SERVER)], env=env)

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"仓库里有一个待修的问题：\n{issue}"}]

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for index in range(self.max_steps):
                reply = self.model.chat(messages, stop=["\nObservation"])   # 1. 问模型
                parsed = parse_response(reply)                              # 2. 解析
                observation, payload = await self._dispatch(                # 3. 调工具
                    session, parsed["tool"], parsed["arguments"])
                messages.append({"role": "assistant", ...})                 # 4. 记回历史
                messages.append({"role": "user", "content": f"Observation: {observation}"})
```

对应到第 0.4 节那张图，四步就是"问→解析→执行→贴回去"。

关于 `async` / `await`：MCP 官方的 Python SDK 是异步的，所以这里用 `asyncio`。
但自检脚本是同步调用的，所以外面包了一层：

```python
def run(self, repo_path: str, issue: str) -> dict:
    return asyncio.run(self._run(repo_path, issue))    # 同步入口，内部跑异步
```

#### 4.5.3 解析器为什么这么啰嗦

模型不会乖乖按格式输出。`parse_response` 和 `_loads_lenient` 里有大量容错代码，
每一段都对应一个真实见过的毛病：

| 模型的实际输出 | 处理办法 |
|---|---|
| 参数外面套了 ` ```json ` 围栏 | 正则先把围栏剥掉 |
| JSON 后面跟一堆废话 | 花括号配对，截出第一个完整的 JSON 对象 |
| 一条回复里把后面几轮也写了 | **按出现位置**判断 Action 和 Final Answer 谁在前 |
| 只写了 Action 没写 Action Input | 当格式错误处理，回一句提示让它重来 |

第三条是我们实际踩的坑，值得展开说：

```python
final = _FINAL_RE.search(text)
action = _ACTION_RE.search(text)
# 谁先出现谁作数：模型常把整条轨迹（多个 Action + Final Answer）一口气写出来，
# 只要第一个 Action 在 Final Answer 前面，就该先执行那个 Action。
if final and (action is None or final.start() < action.start()):
    return {"kind": "final", ...}
```

最初的写法是"只要文本里有 `Final Answer` 就当成完成"。结果模型一次性把
"读文件 → 改文件 → 跑测试 → Final Answer"整条轨迹全写出来了（因为系统提示里的示例
写得太连贯，它照抄了），而解析器看到 `Final Answer` 就以为它干完了，
**三个工具一个都没执行**，直接返回"测试通过：False"。

所以现在改成比位置：第一个 Action 在 Final Answer 前面，就先去执行那个 Action。

同样的道理，写回历史时也只保留第一轮：

```python
messages.append({"role": "assistant", "content": _trim_to_first_round(reply)})
```

#### 4.5.4 停机条件（M4 的核心）

DoD 明确要求"停机条件必须合理，**不能是'步数到了硬停'**"。我们有三条停机路径：

```python
# 路径一：模型自己说完成了（带守卫，见下）
if parsed["kind"] == "final":
    ...
    trace["done_reason"] = f"模型声明完成：{parsed['answer'][:120]}"
    break

# 路径二：测试真的全绿了（最硬的信号）
if payload is not None and payload.get("passed") is True:
    trace["tests_passed"] = True
    trace["done_reason"] = "run_tests 全绿"
    break

# 路径三：连续解析失败，再转下去也是浪费
if parse_fails >= MAX_PARSE_FAILS:
    trace["done_reason"] = "连续解析失败，提前停机"
    break

# 兜底（不算正常停机）：步数用完
else:
    trace["done_reason"] = f"达到步数上限 {self.max_steps}（兜底，非正常停机）"
```

**路径一是最不可靠的**，因为我们实测发现模型会撒谎。第一次运行时它第 0 步就输出：

> Final Answer: 已修改 calculator.py 第 12 行，将 `return a + b` 替换为 `return a - b`

它一次工具都没调用，文件根本没动，而且连改的方向都写反了。
所以现在对 `Final Answer` 加了两道守卫：

```python
if not trace["steps"] and final_rejects < 2:
    nudge = "你还没有读过任何文件，也没有调用过任何工具，不能凭空断定改好了。请先调用 read_file。"
elif last_run_tests_passed is False and final_rejects < 2:
    nudge = "上一次 run_tests 仍然是失败的，不能就此收工。"
```

翻译成人话：**"你一次工具都没用过，凭什么说修好了？"** / **"上次测试还是红的，别急着收工。"**
把这句话当成 Observation 喂回去，模型下一轮通常就去干活了。

最后，收尾时**不看模型自述，只信真实结果**：

```python
trace["tests_passed"] = _pytest_passed(repo)      # 自己跑一遍 pytest 说了算
trace["patch"] = _collect_patch(repo, before)     # 自己算 diff
```

#### 4.5.5 trace 怎么攒出来的

`steps` 里每一步记四样东西：

```python
trace["steps"].append({
    "index": index,                                  # 第几步
    "thought": parsed.get("thought", ""),            # 模型的想法
    "tool_call": {"name": ..., "arguments": ...},    # 它要调什么工具、什么参数
    "observation": observation[:OBS_MAX_CHARS],      # 工具返回了什么（超长截断）
    "raw": reply[:600],                              # 模型原始输出，调试用
})
```

**为什么要记 `raw`？** 因为出问题时你最想知道的是"模型原话到底写了什么"。
我们定位那个"一次性吐出整条轨迹"的 bug，靠的就是翻 trace 里的 `raw` 字段。

`_collect_patch` 优先用 `git diff`，没有 git 就退回"改动前快照 vs 现在"的比对
（`difflib.unified_diff`）——因为 toy-repo 的 git 初始化可能失败（新机器没配 git 身份），
不能假设一定有 git 可用。

---

## 5. 自检脚本怎么判分

`eval/run.py` 里有四个测试函数，每个返回一个 dict，`pass` 字段三态：
`True` 通过 / `None` 跳过 / `False` 失败。

| 测试 | 怎么判 |
|---|---|
| `mcp_server_lists_tools` | `from src.mcp_server import list_tools`，数返回值 ≥5 且每项有 `name` |
| `skill_loader_metadata` | 用 `src/skills` 构造 `SkillLoader`，每个 skill 必须有 name + description，且 ≥2 个 |
| `toy_repo_patch` | **先**从 `calculator.py.orig` 恢复出 bug 版本，**再**调 `agent.run()`，最后自己跑 pytest 看返回码 |
| `swebench_lite_sample` | 没下 parquet 就直接跳过（不 import pandas，也不 import agent） |

注意第三条里的顺序：**每次都是从 bug 版本重新开始**。如果你手动把 bug 修好了再跑自检，
它测的就不是你的 agent 而是"文件本来就是好的"，所以自检帮你恢复了。

还有一点：**判分用的 pytest 是自检自己跑的**（`subprocess.run([sys.executable, "-m", "pytest", "-q"])`），
不是看 agent 自己报告的 `tests_passed`。这是刻意的——防止 agent 自说自话。

---

## 6. 一次完整运行长什么样

真实运行（1.5B Q4 + llama.cpp，共 21 秒）的逐段解读：

```
[step 0] read_file :: 先读一下 calculator.py 里 add 的实现
    obs: {"ok": true, "path": "calculator.py", "total_lines": 11,
          "content": "   1 | def add(a, b):\n   2 |     \"\"\"Return the sum...\n   3 |     return a - b\n..."}

[step 1] edit_file :: add 的实现里把加号写成了减号，改回来
    obs: {"ok": true, "path": "calculator.py", "replaced": 1}

[step 2] run_tests :: 跑测试确认改动
    obs: {"ok": true, "returncode": 0, "stdout": "...  3 passed in 0.01s\n",
          "stderr": "", "passed": true}

===== 结果 =====
tests_passed : True
done_reason  : run_tests 全绿
steps        : 3   调用 3 次模型  3031 tokens  21s
skills_used  : ['test-runner']
patch 行数   : 12
```

几个可以观察的点：

- **第 0 步它先读文件**，没有上来就改——这是系统提示里"第一轮应该是 read_file"起的作用；
- **第 1 步的 `old_str` 精确匹配成功**（`"replaced": 1`），说明它老老实实照抄了原文
  （包括 4 个空格的缩进）；
- **第 2 步跑测试**，`"passed": true` → 触发路径二停机，**没有再问模型一次**。省了一轮，
  也避免了模型反悔；
- `skills_used: ['test-runner']`——issue 里有"pytest"字样，命中了 test-runner 技能包，
  它的正文被塞进了系统提示。

完整的 trace（含每一步的 `raw` 原始输出）见 [`traces.md`](traces.md)。

---

## 7. 把成果存进 git

git 的完整入门（`status` / `add` / `commit` / `push` / fork）写在任务五那份
[`../../task-5-tool-agent/docs/walkthrough.md`](../../task-5-tool-agent/docs/walkthrough.md) 的第 6 节，
这里只说任务六**该提交哪些**：

```bash
cd "D:/新生任务/llm-beginner"

# 看一眼现在的状态
git status

# 加入本次任务的成果
git add task-6-coding-agent/src/          # 你的实现（含 skills/）
git add task-6-coding-agent/REPORT.md
git add task-6-coding-agent/docs/         # 这两份文档
git add task-6-coding-agent/eval/run.py   # 只有你改过自检脚本才需要

git commit -m "task-6: Mini Coding Agent（MCP 工具层 + Skill + agent loop）"
```

**不用管、也不该提交的东西**（`.gitignore` 已经帮你挡住了）：

| 路径 | 为什么不提交 |
|---|---|
| `data/toy-repo/` | 用 `download.py` 随时能重新生成 |
| `models/` | 1.1GB 的模型权重，不该进代码仓库 |
| `eval/result.json` | 自检产物，每次都变 |
| `__pycache__/` | Python 自动生成的缓存 |

> 想确认 `.gitignore` 生效了，跑 `git status --short`，看到 `??` 开头的才需要管，
> 上面这些应该压根不出现。

---

## 8. 自己动手改（最有效的学习方式）

按难度从低到高：

```bash
# ① 关掉 Skill 看看差别（两行就够）
#    CodingAgent(use_skills=False) —— 观察模型"改完不验证"的情况是不是变多了
"D:/新生任务/.venv/Scripts/python.exe" "D:/新生任务/.tools/run_agent.py" --no-skills

# ② 把步数上限压到 2，看它在不够用时怎么表现
"D:/新生任务/.venv/Scripts/python.exe" "D:/新生任务/.tools/run_agent.py" --steps 2

# ③ 给 toy-repo 换个更难的 bug（比如把 factorial 的 range(2, n+1) 改成 range(2, n)）
#    这个 bug 测试里只有第三个用例会挂，模型得看懂失败信息才能定位

# ④ 加一个新工具：比如 list_files（列目录），补进 TOOL_SPECS 和 MCP 注册两处
#    体会一下"工具表是唯一事实来源"这个设计

# ⑤ 加一个新 Skill：比如 bug-locating（"当测试失败需要定位是哪一行代码导致时加载"）
#    写好后跑 ①，看它会不会被命中

# ⑥ 换更大的模型对比：把 3B 的 gguf 下下来，改 llama-server 的 -m 参数即可，
#    agent 代码一行不用改（因为模型端点是通过 OPENAI_BASE_URL 解耦的）
```

最有价值的练习是 **③**——换个真需要动脑的 bug，你会立刻看出 1.5B 模型的极限在哪。

---

## 9. 常见坑（我们实际踩过的）

| 现象 | 原因 | 怎么办 |
|---|---|---|
| `AttributeError: 'Server' object has no attribute 'list_tools'` | 官方 MCP SDK 2.x 改了 API：`FastMCP` 改名 `MCPServer`，低层 `Server` 也换了注册方式 | 用 `from mcp.server.mcpserver import MCPServer`，或用 `add_request_handler` |
| 模型一条回复里把整条轨迹全写出来 | 系统提示里的示例写成了连续一大段，模型照抄 | 示例要**按"第 N 轮"标注**，并明说"一轮只输出一组然后停下" |
| 模型没调工具就说"已修复" | 小模型的抢答倾向 | 加守卫：没调用过工具 / 上次测试还红着，都不认它的 Final Answer |
| 模型自己编 `Observation:` | 它看到提示词里有这个格式就顺着往下写 | 生成时 `stop=["\nObservation"]` 强行截断 |
| `ModuleNotFoundError: No module named 'mcp'` | 用了 Anaconda base 的 python，那个环境没装 | 一律用 `D:\新生任务\.venv\Scripts\python.exe` |
| MCP 客户端连不上，报 `Connection closed` | server 里往 stdout 打了调试信息，污染了 JSON-RPC | 调试输出一律 `file=sys.stderr` |
| `llama-server` 起不来或极慢 | 模型文件路径含中文 / 线程数设太高 | 路径用引号包住；`-t` 用物理核数（这台机器是 4） |
| agent 改了测试文件让测试"通过" | 模型走捷径 | ISSUE 里明确"不要改测试文件"，Skill 里也写死；真要防得靠工具层拦截写 `test_*.py` |

---

## 10. 术语小抄

| 词 | 一句话解释 |
|---|---|
| **Agent** | 能自己决定"下一步做什么"、并真的去做（调用工具）的程序 |
| **Agent loop** | 模型输出动作 → 程序执行 → 结果回灌 → 再问模型，这么转圈 |
| **ReAct** | 一种让模型"先说想法（Thought）再说动作（Action）"的提示格式 |
| **MCP** | Model Context Protocol，把工具做成标准服务的协议，本任务用 stdio 传输 |
| **stdio** | 用标准输入输出通信；⚠️ stdout 被协议占用，调试信息只能走 stderr |
| **Tool / 工具** | 一个具体动作，如 `read_file`。本任务有 6 个 |
| **Skill / 技能包** | 一段"遇到这类活该怎么干"的流程说明，按需加载（渐进式披露） |
| **Subagent / 子 agent** | 有独立上下文的助手，主 agent 只看它的结论（本任务未实现） |
| **渐进式披露** | 平时只暴露一句话描述，真要用时才把完整内容读进来 |
| **Observation** | 工具执行结果，被贴回对话让模型看到 |
| **Trace** | 整轮运行的完整记录（每步的想法、动作、结果），出问题时靠它排查 |
| **停机条件** | 什么时候算干完了。必须是硬信号（测试通过），不能听模型自己说 |
| **上下文窗口** | 模型一次能"看到"的 token 上限，本任务设成 8192 |
| **token** | 模型处理文本的最小单位，中文大约一个字 1-2 个 token |
| **量化（Q4_K_M）** | 把权重压成 4 位存储，体积缩到约 1/4，代价是精度略降 |
| **GGUF** | llama.cpp 用的模型文件格式，权重 + 量化信息打包在一起 |
| **shell 注入** | 把用户/模型给的字符串拼进 shell 命令导致被注入执行；用列表参数 + `shell=False` 防 |
| **路径穿越** | 用 `../../` 绕过目录限制读到外面；`resolve()` 之后再检查前缀来防 |
| **patch / diff** | 描述"改了哪些行"的文本，unified diff 是标准格式 |
