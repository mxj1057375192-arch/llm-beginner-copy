# 任务五实验报告：工具调用 Agent

## 一、自检结果

```
[通过] tools_individual:      {"pass": true, "results": {"calculator": true, "python_sandbox": true,
                               "file_search": true, "wiki": true}, "network_skipped": null}
[通过] multi_tool_success_rate: {"pass": true, "rate": 0.7, "n": 10}
[跳过] error_recovery:          {"skip": "需要学生实现 inject_error 测试钩子；可选实验"}
```

完整结果见 [eval/result.json](eval/result.json)。10 题逐题：

| 题 | 结果 | 最终答案 |
|---|---|---|
| 1 计算 + 位数 | ✅ | `(123 + 456) * 789 = 456831，共 6 位` |
| 2 质数求和 | ✅ | `1060，共 10 位` |
| 3 数 .md 文件 | ✅ | `2` |
| 4 图灵机发明者 | ✅ | `图灵机的发明者是艾伦·图灵。` |
| 5 Hinton 出生年 + 年龄 | ✅ | `Geoffrey Hinton 出生年份是 1947 年，到 2026 年他已经过去了 79 年。` |
| 6 回文判断 | ❌ | `level: 100，world: 100`（编的，没调用工具） |
| 7 含 TODO 的文件路径 | ❌ | 只说"找到 1 个文件"，漏了路径 `todo_note.md` |
| 8 sqrt 精度 | ✅ | `sqrt(2026) ≈ 45.011110（小数点后 6 位）` |
| 9 Transformer 年份 + 年数 | ❌ | `13579，共 5 位`（复读了 prompt 示例里的字面量） |
| 10 文件第一段 | ✅ | `这是任务五的本地文件检索测试文件。` |

命中率 7/10 = 0.7，刚过 0.6 的线。**注意波动很大**：迭代过程中每轮全量自检的总分在 5/10 — 7/10 之间，失败集中在第 2、6、7、9 题，同一份代码重跑这几题时好时坏（第 5、8 题稳定通过，第 6、7、9 题尤其看运气）。

**DoD 勾选**

- [x] **M1** 4 个工具（calculator / python_sandbox / file_search / wiki），各带 `TOOL_SCHEMA` 与 `run(args)`，自检 `tools_individual` 通过
- [x] **M2** 手写 ReAct 循环（Thought / Action / Action Input / Observation），含工具路由、步数上限（5 步）、Final Answer 终止
- [x] **M3** 工具异常捕获后把错误消息塞回 Observation 让 agent 自我纠错，单次失败不 crash 整个循环
- [x] **M4** `multi_tool_success_rate` 通过（0.7 > 0.6）
- [ ] S1–S4 加分项均未做

## 二、实现

| 文件 | 内容 |
|---|---|
| [src/tools/calculator.py](src/tools/calculator.py) | AST 白名单求值（不用 eval），四则运算 + `math` 函数；输出带整数位数与 6 位小数 |
| [src/tools/python_sandbox.py](src/tools/python_sandbox.py) | 受限 exec：白名单 builtins / import、`sys.settrace` 行数预算掐死循环、stdout 捕获（**教学级防护，见文件头警告**） |
| [src/tools/file_search.py](src/tools/file_search.py) | 先按文件名（子串 + fnmatch）、再按内容（正则/子串）检索，回带内容片段；`dir` resolve 后校验必须落在任务目录内 |
| [src/tools/wiki.py](src/tools/wiki.py) | zh 走 TextExtracts、en 走 `action=parse&section=0`，精确标题查不到就 `list=search` 兜底；带重试、中英互备、REST 摘要接口兜底 |
| [src/agent.py](src/agent.py) | 手写 ReAct 循环（核心循环约 60 行，其余是解析容错、收尾自检与注释，共 430 行） |
| [src/llm.py](src/llm.py) | 模型薄封装：默认进程内 transformers 加载，设 `OPENAI_BASE_URL` 则改走 OpenAI 兼容 API |

循环里几个必要的设计（都是实测踩出来的，细节见注释）：

1. **用 `Observation:` 作停止串**。0.5B 会顺着 few-shot 的格式自己把 Observation 编出来（实测见过它写 `Observation: 2500`），不掐掉的话 agent 自问自答，真工具一次都不会被调用。
2. **工具异常 → Observation 文本**（M3）。`KeyError` 单独处理：直接把"缺哪个参数、该怎么补"写进错误消息。
3. **Action 解析阶梯**：严格 JSON → 首个配平花括号块 → `ast.literal_eval` → 正则硬抠 `"key": "value"` → 单参数工具按裸值兜住；再不行回一句格式提示让模型重写。
4. **收尾自检**：一次工具都没调成、或题目要算数字却没成功调用过计算工具时，拒绝收尾并要求补一轮。
5. **判重只拦"上次成功"的重复调用**。工具失败后用同样参数重试是正确行为，早期版本把这种重试也拦了，trace 里抓到的（见 [docs/traces.md](docs/traces.md) 错误恢复一节）。

完整 trace 见 [docs/traces.md](docs/traces.md)：单工具调用、两次注入错误（一次改对参数恢复、一次没能恢复）、wiki→calculator 两步、文件检索回带片段。

## 三、模型与运行配置

- 模型：**Qwen2.5-0.5B-Instruct**（复用任务四已下载的权重，本机无 CUDA，纯 CPU 4 线程）
- 解码：贪心（`do_sample=False`），`max_new_tokens=512`，`max_steps=5`
- 速度实测：单步 15–30 秒，单题 30–120 秒，**10 题全量一轮 10–15 分钟**（含模型加载）

README 建议用 7B。本机无 GPU，7B 量化在 4 核 CPU 上约 2 tok/s，单轮自检要 1–2 小时，迭代成本太高；因此选 0.5B 跑通流程，把"更大模型"留作 S2（未做）。这也意味着本报告的失败案例基本都是**模型能力**问题，不是循环逻辑问题。

## 四、实验观察（约 500 字）

**格式不是瓶颈，参数和自纠错才是。** 0.5B 一次 few-shot 就学会了 ReAct 的四段格式，Thought 也说得像模像样；真正的失败集中在两类：① 参数写不对（把 `math.sqrt(2026)` 交给只认 `sqrt` 的工具、把整个 JSON 当 `code` 的值再套一层、漏参数）；② 失败后不修正，而是**凭印象编答案**。实测它编过一整段不存在的英文 README 段落，也编过"100 以内质数是 2,3,5,7，和为 20"这种一眼假的结论。所以"关键词命中率"这个判分口径是被逼出来的：只看自报的 success，成绩会虚高得离谱。

**few-shot 污染比想象中严重。** 我在示例里写过 `3 年`，模型答第 9 题就写"过了 3 年"（正确答案是 9）；我为了说明"答案要带数字"在规则里写了 `456831，共 6 位`，模型被追问时直接把这一串复读成第 9 题的答案。教训是示例里的实体和数字必须与评测题无关，且**任何字面量都可能被当成答案搬走**。

**工具输出的格式直接决定成败。** 第 8 题要求小数点后 6 位，而 `math.sqrt` 的原样 repr 是 `45.0111097397076`，模型照抄过 `45.011109`（少一位），两种判定关键词都不命中；把 `45.011110` 挪到输出最前面、并附上 6 位小数值后，这题才稳定通过。同理第 6 题要求字面 `True`/`False`，模型习惯输出中文"是回文/不是回文"，我加了"把对象和结果一起写出来"的例子才好转（但仍会波动）。

**"通过"不等于"工具真被用上"。** wiki 偶发 TLS 断连（本机只能经系统代理访问维基），这时 0.5B 会用预训练记忆补答案——1947、2017 这种常识恰好是对的。第 4、5、9 题的答案因此可能来自模型记忆而非工具返回，这正是任务说明里"不信任 agent 自报 success、按答案关键词校验"要防的情况，但也说明关键词校验同样防不住记忆补答案。

**最有效的一处改动**：把 `max_new_tokens` 从 256 提到 512。`python_sandbox` 的 `code` 是一整段代码，JSON 里还带 `\n` 转义，256 个 token 会在 JSON 中途截断——模型收到的报错是"语法错误"，它根本不知道自己被截了，于是原地打转。第 2 题反复失败好在此处。

## 五、踩到的坑

1. **prompt 模板里写了 `{{"参数名": "参数值"}}`**（本想用 `.format()` 的转义），后来改用 `.replace()` 却忘了改回单花括号。模型逐字符照抄，所有 `Action Input` 都退化成 `{}`。小模型对格式模板是逐字符模仿的。
2. **`ast.literal_eval` 不只抛 ValueError/SyntaxError**，遇到被改写的字面量（如 `{{"a": 1}}` 被当成 set 里的 dict）会抛 TypeError。没兜住会让整个 `run` 崩掉，与 M3 冲突。
3. **中文条目的标题和书面叫法不一致**：评测题里的「Transformer (机器学习模型)」在中文维基上的实际标题是「Transformer架构」，精确标题查询直接 missing，必须用 `list=search` 兜底。
4. **英文维基没有 TextExtracts 扩展**：`prop=extracts` 在 en 上返回空串，得改用 `action=parse&section=0&prop=text` 再剥标签；所以用 `wikipedia-api` 这类封装库反而不好使，直接 requests 打 `w/api.php` 更可控。
5. **维基的连通性靠系统代理**：本机 `requests` 会读 Windows 注册表里的代理（`127.0.0.1:7890`），实测经代理 4/4 通、直连 0/4 通，且有约 10–25% 的间歇性 TLS 断连，调用方必须自带重试。
6. **判重逻辑别把正确行为拦掉**：只应拦"上次成功"的重复调用。早期版本连失败后的原参数重试一起拦了，注入错误的那条 trace 里模型想重试却被顶回去，最后放弃作答。
7. Windows 控制台默认 GBK，跑临时脚本要 `PYTHONIOENCODING=utf-8`（仓库自带的 `_eval_harness.setup_console()` 已经处理了自检路径）。

## 六、未做的部分

- S1 Qwen-Agent 对照、S2 模型尺寸对比（1.5B/7B）、S3 prompt 模板消融、S4 `inject_error` 跑通 `error_recovery` —— S4 的钩子已在 `src/agent.py` 里预留（`self.inject_error`，返回 `None` 表示该步不注入），但 `eval/run.py` 里对应的测试仍是跳过状态，没去改自检脚本。
- 多步任务的上下文压缩（词元预算管理）、并行工具调用、prompt injection 防御均未做。

## 七、复现

```bash
python data/download.py    # 生成 data/tasks.json（10 题）与 data/agent-fixtures/
python src/demo.py --task 3   # 跑单题、打印全过程（不打分，供观察）
python eval/run.py         # 自检，结果写 eval/result.json
```

第一次接触这个任务的话，先看 [docs/walkthrough.md](docs/walkthrough.md)（小白向：环境、代码逐行、git 怎么用）。

模型默认从 `AGENT_MODEL` 找本地 Qwen2.5-Instruct，未设时依次尝试本任务 `models/`、任务四的 `models/`。设 `OPENAI_BASE_URL` 可改走 Ollama / vLLM 的 OpenAI 兼容接口；设 `AGENT_VERBOSE=1` 可打印每步 Thought / Action / Observation。
