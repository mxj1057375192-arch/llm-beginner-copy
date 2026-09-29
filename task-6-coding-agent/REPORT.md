# 任务六实验报告：Mini Coding Agent

> 配套文档：[`docs/walkthrough.md`](docs/walkthrough.md)（小白向手把手指南，含逐文件拆解与常见坑）、
> [`docs/traces.md`](docs/traces.md)（完整运行 trace 与两个反例）。

## 一、自检结果

```
[通过] mcp_server_lists_tools: {"pass": true, "tools": ["read_file", "write_file",
                                "edit_file", "run_tests", "git_diff", "git_apply"]}
[通过] skill_loader_metadata:  {"pass": true, "count": 3, "missing_meta": []}
[通过] toy_repo_patch:         {"pass": true, "tests_passed": true, "trace_has_patch": true}
[跳过] swebench_lite_sample:   {"skip": "data/swebench-lite-sample.parquet 不存在"}
```

完整结果见 [eval/result.json](eval/result.json)。连跑 3 次结果一致（`temperature=0`）。

**DoD 勾选**

- [x] **M1** 手写 MCP server，暴露 6 个工具；`python src/mcp_server.py` 可独立以 stdio 起服务，
      标准 MCP client 能握手、枚举、调用；模块顶层导出 `list_tools()`
- [x] **M2** `SkillLoader` + 3 个带 YAML front-matter 的 `SKILL.md`，自检 `skill_loader_metadata` 通过
- [x] **M3** `CodingAgent.run(repo_path, issue) -> Trace`，在 toy-repo 上修好 `calculator.add`
      并让 `python -m pytest` 全绿（3 passed）
- [x] **M4** trace 逐步记录 thought / tool_call / observation，含 `steps` / `patch` / `tests_passed`
- [ ] S1–S4 加分项未做；**Subagent 层未实现**（见「六、未完成」）

## 二、实现

| 文件 | 内容 |
|---|---|
| [src/mcp_server.py](src/mcp_server.py) | 6 个工具（read_file / write_file / **edit_file** / run_tests / git_diff / git_apply）。路径先 `resolve()` 再校验落在 repo 内；subprocess 一律 list 形式 + 限 cwd + 超时；git 只放行 diff / apply，`git_apply` 先 `--check` 预检、失败不落盘；工具异常全转成结构化 `{"ok": false, "error": ...}` |
| [src/skill_loader.py](src/skill_loader.py) | 约 60 行的渐进式披露：`list_skills()` 只读 front-matter，`match()` 按 description 与任务文本的词重叠挑，`load()` 命中后才取正文 |
| [src/skills/](src/skills/) | `test-runner` / `code-review` / `pr-description-writer`，每个含 name + 写清「何时加载」的 description |
| [src/agent.py](src/agent.py) | ReAct 循环 + 容错解析 + trace 收集（核心循环约 70 行，其余是解析容错与停机守卫） |
| [src/llm.py](src/llm.py) | OpenAI 兼容端点的薄封装，带重试与 token 统计 |

## 三、环境与模型

本机**无 GPU**（i5-1135G7，4 核 8 线程，16 GB 内存），任务点名的 Qwen2.5-Coder-7B-Instruct
FP16 约 15 GB 装不下、Q4_K_M 在 4 核上约 2 tok/s，一轮对话要几十秒，不适合反复迭代。

实际采用 **Qwen2.5-Coder-1.5B-Instruct Q4_K_M**（1.1 GB，ModelScope 下载）+
**llama.cpp b11205 CPU 版**（`llama-server`，`-c 8192 -t 4`），实测生成 **25 tok/s**，
比 7B 量化快一个数量级。选它是因为 toy-repo 的 bug 足够简单，实测一次通过；
模型端点由 `OPENAI_BASE_URL` 指定，换 3B / 7B 不需要改 agent 代码。

## 四、踩到的坑

1. **官方 MCP SDK 2.x 改了 API**：`FastMCP` 更名 `MCPServer`（`mcp.server.mcpserver`），
   低层 `Server` 也不再提供 `@server.list_tools()` 装饰器，改为 `add_request_handler`。
   按 0.9/1.x 的写法会直接 `AttributeError`。
2. **prompt 示例写成一整段导致模型把整条轨迹一次吐完**：系统提示里的三轮示例是连续文本，
   1.5B 模型直接照抄，一条回复里写了 3 个 Action 加 Final Answer。改成按「第 N 轮」标注
   并明说「一轮只输出一组然后停下」后才正常。解析器也相应加固：**按出现位置**判断
   Action 与 Final Answer 谁在前，而不是优先匹配 Final Answer。
3. **小模型会抢答**：第一次运行模型在第 0 步就编了一句「已把 `a + b` 改成 `a - b`」的
   Final Answer（方向还写反了），一次工具都没调。加了守卫：没调用过工具、或最近一次
   `run_tests` 仍是失败的，都不接受 Final Answer，把提示作为 Observation 喂回去。

## 五、实验观察

**工具粒度比模型大小更影响成败。** 最初只有 `write_file`（整文件覆盖），对 1.5B 模型来说
要把 11 行文件转义进 JSON 字符串，很容易在 `\n` 上出错；换成 `edit_file`（只给 old_str /
new_str 两个片段）后一次就过。`edit_file` 还加了「old_str 必须唯一命中」的检查——模型很
容易只给 `return a - b` 这种片段，真让它替换就可能改错地方，命中不唯一时直接拒绝比默默改
错安全。

**显式停机信号必须由代码判定，不能信模型自述。** 模型三次运行里两次主动声称「已修复」，
实际文件没动。最终把停机条件定为「`run_tests` 的结构化返回值里 `passed` 为真」，并且收尾
时另跑一次真实 pytest 覆盖 `trace["tests_passed"]`，不采信模型的话。toy-repo 的修复稳定在
**3 步 / 约 3000 tokens / 21 秒**内完成。

**渐进式披露在弱模型上是刚需而非优化。** issue 文本里「跑测试」的字样会命中 `test-runner`
skill，把测试诊断流程塞进系统提示后，模型「改完不验证就收工」的概率明显下降。

## 六、未完成

- **Subagent 层**：本任务能力三层栈的顶层未实现，`src/subagents/` 为空。当前主 agent 只有
  Tools + Skills 两层。补的话计划做「代码搜索」和「测试执行」两个独立 context 的子 agent
  （各自独立 message 列表、独立步数上限、工具子集，主 agent 只看摘要）。
- **S1–S4 加分项**：均未做。S1（量化 vs FP16）需要先解决 7B 在 CPU 上的可跑性；
  S4 需要 `--with-swebench` 并自行 clone 对应仓库，本机下载与算力都吃紧。
