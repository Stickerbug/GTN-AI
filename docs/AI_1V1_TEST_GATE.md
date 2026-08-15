# Phelren V1 账号对局

## 当前实现

入口位于多人游戏的 `1v1` 页面，是独立按钮，不是第五种匹配模式。服务器启用后，所有
登录账号均可看到；游客无法通过界面或直接发送 Socket 事件开始对局。它会创建一局
不计统计、不计花阶分的本地 1v1：主游戏进程仍以生产 `GameEngine` 作为唯一规则来源，
AI 模型和搜索运行在 `GTN-AI\.venv` 的独立 Python 3.12 子进程中。

AI 子进程只监听随机本机端口 `127.0.0.1`，每次启动使用随机令牌。主游戏不导入
PyTorch，AI 进程退出或失败不会切换正式匹配规则。当前默认策略是：

```text
structured-v2-search-dagger-v2.epoch-06.pt
+ candidates=3, rollouts=2, horizon=2, belief=true
```

实测首次加载约 4.2 秒，首步完整进程往返约 1.9 秒，其中搜索约 1.1 秒。具体耗时随
局面和电脑负载变化。

## 本地开启方法

在启动本地游戏服务前设置：

```powershell
cd "E:\Garden of Thorn 荆棘花园\Python联机版"
$env:GTN_AI_1V1_TEST_ENABLED = '1'
$env:GTN_PORT = '5099'
python app.py
```

登录账号进入大厅后，在多人游戏的 `1v1` 页可以看到“对战 Phelren V1”。AI 在所有
语言中均显示为 `Phelren`；未登录玩家和未设置环境变量的服务不会显示入口。特别致谢
开关与此入口完全无关。

## 服务器部署

游戏服务通过回环 HTTP 调用独立子进程，PyTorch 不进入主游戏进程。推荐目录：

```text
/opt/gtn-release/                 游戏服务
/opt/GTN-AI/                      GitHub 源码、venv 与模型
/var/lib/gtn-ai/human-sessions/   持久诊断数据
/etc/gtn/ai.env                   systemd 环境配置
```

在游戏仓库执行 `scripts/setup_gtn_ai.sh` 可克隆/更新
`https://github.com/Stickerbug/GTN-AI.git`、安装 CPU 版 PyTorch 并生成环境文件。模型权重
受 `.gitignore` 排除，不会随 GitHub 仓库到达服务器，必须单独上传或通过私有 Release、
OSS 地址提供 `GTN_AI_MODEL_SOURCE`。然后为正式服务加入：

```ini
[Service]
EnvironmentFile=-/etc/gtn/ai.env
```

默认最多同时存在 2 局未结束的人机对局，重型决策只并行 1 个；入口满载时会明确提示稍后
重试。worker 最多缓存 8 个会话策略，结束结果页不占用对局容量。诊断保留 14 天且总量
限制为 2 GiB，线上不再为每局额外生成一份重复 ZIP。以上数值均可在环境文件中调整。

测试局只使用玩家当前选择的官方模组；娱乐、DLC、社区和故事模式内容会被排除。双方
均经过正式 1v1 的配装倾向选择、倾向公开和选牌流程。真人自行选择倾向与完整牌组；
Phelren 目前从生产引擎给出的合法候选中自动选择。进入战斗后沿用正式 1v1 的界面和
操作，只把会话与 AI 推理隔离在本机，不显示训练场的改牌、设置下次抽牌或撤销功能。

## 诊断数据

本地默认保存到：

```text
GTN-AI\.runtime\live\human-sessions\<session-id>\
```

服务器由 `GTN_AI_DIAGNOSTICS_ROOT` 改为持久目录。内容包括：

- `manifest.json`：模型、规则指纹、官方模组、座位与结局
- `decisions.jsonl`：每一步公开观测、合法动作、实际动作和搜索摘要
- `snapshots\*.pkl.gz`：该步执行前的完整私有引擎快照
- `markers.jsonl`：玩家通过“标记此步”记录的可疑决策
- `exports\*.gtnai.zip`：仅在 `GTN_AI_EXPORT_FINISHED=1` 时生成的完整诊断包

私有快照只应在可信机器读取。它使用 Pickle，并且包含双方隐藏牌信息，不能上传到公开
接口或交给不可信程序反序列化。worker 会在落盘前把玩家昵称替换为通用座位名；诊断
元数据不写账号 ID、昵称或 IP。

将已结束测试局转换为严格玩家数据：

```powershell
cd "E:\Garden of Thorn 荆棘花园\GTN-AI"
.\.venv\Scripts\python.exe -m gtn_ai.diagnostic_data `
  .runtime\live\human-sessions `
  --output runs\human-vs-ai.jsonl.gz
```

默认只导入真人、未被标记且确实属于当时合法动作集的决策。报告会分别列出未结束、
非法、未能规范化、被标记及非真人动作的数量。需要同时研究 AI 动作时可加
`--actors human,ai`；只有在人工核验后才建议加 `--include-marked`。

### 高预算重标注

普通导入保留玩家实际选择，适合行为分析；要把局面变成更可靠的教师标签，应从执行前
快照重新搜索：

```powershell
cd "E:\Garden of Thorn 荆棘花园\GTN-AI"
.\.venv\Scripts\python.exe -m gtn_ai.diagnostic_relabel `
  .runtime\live\human-sessions `
  --output runs\human-vs-ai-teacher.jsonl.gz `
  --only-candidates `
  --trust-private-snapshots
```

默认同时重算真人和 AI 决策，使用比实时 AI 更高的候选数、rollout、置信度自适应预算
和搜索深度。只重算玩家标记的关键局面可加 `--only-marked`；由于人通常会在发现问题
后一两步才点击标记，该模式默认同时复核标记步及其前 2 步，可用
`--marker-lookback` / `--marker-lookahead` 调整。窗口内的动作不会直接被判错，仍由离线
教师判断是纠错样本还是保持原策略的锚点。玩家没有标记时，推荐使用
`--only-candidates`：它会自动选择实时搜索改变基础动作、实时搜索分差较小的局面，并按
确定性采样保留 10% 锚点；可用 `--uncertainty-margin` 和 `--anchor-rate` 调整。试跑可
加 `--max-decisions 10`。输出保留原
动作，并在每步 `teacher` 中加入搜索动作分布、价值和分差。

不要按单个决策随机划分少量真人数据。同一局相邻状态高度相关，应先按完整诊断会话
和官方模组组合划分：

```powershell
.\.venv\Scripts\python.exe -m gtn_ai.hard_examples `
  runs\human-vs-ai-teacher.jsonl.gz `
  --train-output runs\human-hard-train.jsonl.gz `
  --validation-output runs\human-hard-validation.jsonl.gz `
  --base-policy "structured:models\structured-v2-search-combined-m05-head-v1.pt" `
  --validation-fraction 0.2 `
  --overwrite
```

筛选器保留高置信教师/基础策略分歧、所有人工标记窗口，以及少量高置信一致锚点；并为
不同用途写入样本权重。训练集与验证集不会共享诊断会话。随后分别构建缓存：

```powershell
.\.venv\Scripts\python.exe -m gtn_ai.build_recorded_teacher_cache `
  runs\human-hard-train.jsonl.gz `
  --config-checkpoint models\structured-v2-contextual-broad-v1.pt `
  --output datasets\human-hard-train-context-v1 `
  --overwrite

.\.venv\Scripts\python.exe -m gtn_ai.build_recorded_teacher_cache `
  runs\human-hard-validation.jsonl.gz `
  --config-checkpoint models\structured-v2-contextual-broad-v1.pt `
  --output datasets\human-hard-validation-context-v1 `
  --overwrite
```

`gtn_ai.train_correction` 的 `--validation-cache` 可直接读取后一份独立缓存，关闭按单条
样本随机切分。纠错微调仍需搭配宽覆盖 replay/锚点，防止少量困难样本破坏模型已经掌握
的正常决策。

重标注器会重新生成公开观测和合法动作；只要与记录的动作顺序、规则指纹或决策玩家不
一致，就拒绝该样本。`--trust-private-snapshots` 是必要的显式确认：Pickle 快照只能来自
可信本机，不能处理玩家上传或来源不明的诊断包。

2026-08-15 的首轮实跑从 5 场带标记真人会话中恢复了 23 个窗口状态，其中 21 个通过
完整一致性检查，2 个因教师无法可靠形成标签而被丢弃。按整场会话划分后为 17 个训练
状态和 4 个独立验证状态。这个规模只用于验证闭环，不能据此训练并替换线上模型。

## 真人数据的训练价值

大量玩家对局有用，但不应直接把所有真人动作当作最优示范。推荐流程是：

1. 优先收集玩家胜负、模型版本、规则指纹、超时和玩家标记的坏步。
2. 从坏步执行前快照恢复局面，以玩家当时相同的可见信息离线扩大搜索预算。
3. 将重算后的动作分布、价值和原动作差异写成教师标签。
4. 混入旧稳定数据蒸馏学生模型，再做独立种子、双座位擂台验证。
5. 只有新学生在启发式、历史冠军和真人盲测上都改善后才替换默认模型。

真人操作可以作为辅助行为先验，尤其适合补充模型很少遇到的选择窗口；但玩家水平、
试验行为和投降会产生标签噪声，因此不能只按胜者动作做无筛选行为克隆。

## 边界

- 当前信念搜索在独立进程内从完整引擎快照构造近似隐藏牌样本，尚未完成正式竞技所需的
  信息边界审计，因此只用于公开但不计分的 Phelren 对局，不能加入真人匹配队列。
- AI 已覆盖 V2 数字、单选、玩家、卡牌和装备的单选/多选窗口，并由生产验证器过滤非法
  候选；未知控件或没有合法值时仍采用生产引擎兜底动作并留下诊断。
- Phelren 对局拥有独立 `P-` 回放，但不参与正式匹配、GR、胜率或奖励。
- 官方规则改变后，规则指纹和独立评测必须重新检查。
