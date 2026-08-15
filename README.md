# GTN-AI

荆棘花园正式 1v1 的独立 AI 训练工程。它直接调用 `Python联机版` 的生产
`GameEngine`，但不会修改或启动游戏服务器。

当前版本提供的是一套可验证的 AI 基础设施，而不是声称已经求得“绝对最优”策略：

- 仅加载 12 个官方模组，排除娱乐、DLC、社区和故事模式内容
- 支持官方模组的全部 4096 个请求组合
- 将配装倾向、倾向子选择、刷新和生产 `6/4/3/2` 十五轮选牌纳入模型动作
- 观测只包含该座位正常可见的信息，不暴露对手手牌或引擎实例 ID
- 覆盖出牌、目标、反制、选牌窗口、排序、聚变选择、装备触发与结束回合
- 提供随机/启发式基线、自我对局、JSONL 轨迹、线性与神经训练、对称擂台评估
- 支持按权重随机配对的联赛自我对局、神经策略温度探索和进程内模型复用
- 支持从兼容 checkpoint 增量训练、按终局胜负加权示范并整局剔除循环恢复轨迹
- 支持带行为概率、GAE 与权重指纹锁的 on-policy PPO，直接按最终胜负改进策略/价值网络
- 支持同一训练策略对镜像、历史冠军和启发式对手的混合轨迹，并自动忽略对手决策
- 神经模型按“公开观测 + 公开历史 + 可变合法动作”逐项打分，并分别训练预对局与战斗头
- 提供结构化 Transformer v2：玩家、状态、装备、牌区、单牌、选择项和历史事件分别编码
- 可将 v8 的完整动作分布与价值缓存为带版本/权重指纹的张量分片，再反复蒸馏 v2
- 提供极轻量 HTTP 推理服务，适合以后作为游戏服务器的本机 sidecar
- 提供面向登录账号的真人对战入口、隔离 AI worker、脱敏决策快照和坏步标记
- 用全局规则指纹阻止旧模型在卡牌或结算规则更新后静默运行
- 可审计旧 `.gtnreplay` 是否足以用于严格行为克隆

## 快速开始

在 PowerShell 中：

```powershell
cd "E:\Garden of Thorn 荆棘花园\GTN-AI"
python -m pytest -q
python -m gtn_ai.validate_loadouts
```

生成压缩的自我对局训练集：

```powershell
python -m gtn_ai.self_play `
  --games 1000 `
  --sample-mod-combinations `
  --policy-0 heuristic `
  --policy-1 heuristic `
  --workers 8 `
  --record-decisions `
  --output runs\selfplay-v4.jsonl.gz
```

从历史模型、探索模型和启发式策略组成联赛（重复条目即提高抽中权重）：

```powershell
.\.venv\Scripts\python.exe -m gtn_ai.self_play `
  --games 3000 `
  --sample-mod-combinations `
  --league-policy neural-explore-cpu:models\variable-action-v4.pt `
  --league-policy neural-explore-cpu:models\variable-action-v4.pt `
  --league-policy neural-cpu:models\variable-action-v4.pt `
  --league-policy neural-cpu:models\variable-action-v3.pt `
  --league-policy heuristic `
  --workers 6 `
  --record-decisions `
  --output runs\league-next.jsonl.gz
```

训练无第三方依赖的行为克隆基线模型（默认目标）：

```powershell
python -m gtn_ai.train_linear runs\selfplay-v4.jsonl.gz `
  --output models\hashed-linear-v1.json `
  --epochs 5
```

如需使用终局胜负回归目标，额外传入 `--objective monte-carlo`。

训练带公开历史记忆的神经策略/价值模型需要 Python 3.12 与 PyTorch：

```powershell
# 首次使用时，用任意 Python 3.12 解释器创建 .venv；本工作区已经创建完成。
& "C:\path\to\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[training,dev]"
.\.venv\Scripts\python.exe -m gtn_ai.train_neural runs\selfplay-v4.jsonl.gz `
  --output models\variable-action-v1.pt `
  --epochs 5 `
  --device auto
```

从当前冠军继续做胜负加权训练：

```powershell
.\.venv\Scripts\python.exe -m gtn_ai.train_neural runs\league-next.jsonl.gz `
  --output models\variable-action-next.pt `
  --init-checkpoint models\champion.pt `
  --epochs 2 `
  --batch-size 256 `
  --learning-rate 0.000075 `
  --value-loss-weight 0.03 `
  --winner-policy-weight 2.0 `
  --loser-policy-weight 0.15 `
  --draw-policy-weight 0.5 `
  --device xpu
```

默认会丢弃曾触发循环恢复的整局轨迹；只有诊断时才应传入
`--include-recovered-episodes`。

从当前模型采集严格 on-policy 轨迹并直接优化胜负：

```powershell
.\.venv\Scripts\python.exe -m gtn_ai.self_play `
  --games 2000 `
  --policy-0 neural-onpolicy-cpu:models\champion.pt `
  --policy-1 neural-onpolicy-cpu:models\champion.pt `
  --sample-mod-combinations `
  --record-decisions `
  --workers 8 `
  --output runs\onpolicy-next.jsonl.gz

.\.venv\Scripts\python.exe -m gtn_ai.train_actor_critic `
  runs\onpolicy-next.jsonl.gz `
  --init-checkpoint models\champion.pt `
  --output models\actor-critic-next.pt `
  --epochs 1 `
  --batch-size 1024 `
  --shuffle-buffer 32768 `
  --gae-lambda 0.95 `
  --device xpu
```

`neural-onpolicy-*` 会记录实际采样分布、旧价值和模型权重指纹。训练器拒绝指纹不匹配、
强制兜底或不完整的轨迹；混合历史对手时只学习当前策略座位上的决策。

构建一次结构化 v2 教师缓存，并从 v8 蒸馏：

```powershell
.\.venv\Scripts\python.exe -m gtn_ai.build_structured_cache `
  runs\onpolicy-v7-mirror-20260814.jsonl.gz `
  runs\onpolicy-v7-v5-p0-20260814.jsonl.gz `
  runs\onpolicy-v7-v5-p1-20260814.jsonl.gz `
  runs\onpolicy-v7-v4-p0-20260814.jsonl.gz `
  runs\onpolicy-v7-v4-p1-20260814.jsonl.gz `
  runs\onpolicy-v7-heuristic-p0-20260814.jsonl.gz `
  runs\onpolicy-v7-heuristic-p1-20260814.jsonl.gz `
  --teacher models\champion.pt `
  --output datasets\structured-v2-v8 `
  --expected-decisions 322744 `
  --device xpu

.\.venv\Scripts\python.exe -m gtn_ai.train_structured `
  datasets\structured-v2-v8 `
  --output models\structured-v2-distilled.pt `
  --epochs 4 `
  --batch-size 256 `
  --shuffle-buffer 2048 `
  --device xpu
```

缓存保存的是结构化稀疏张量、v8 对每个合法动作的 logit 和价值，不是再次复制 JSON。
它会校验轨迹、规则、观测、动作、特征和教师权重指纹。两个命令默认每 10 秒在
`stderr` 更新进度与吞吐；总量已知时还显示 ETA。可用 `--expected-decisions` 仅提供缓存
进度总量，用 `--progress-interval` 调整频率，或用 `--quiet` 关闭。
蒸馏模型可通过 `structured:CHECKPOINT`、`structured-cpu:CHECKPOINT` 用于自我对局和擂台。

离线 belief rollout 教师可用以下策略规格运行：

```powershell
--policy-0 "unsafe-rollout-cpu:models\structured-v2-search-dagger-v2.epoch-06.pt;candidates=3;rollouts=2;max-rollouts=8;confidence=.075;batch=2;horizon=8;belief=true;annotate=true"
```

- `belief=true` 为每次 rollout 重新采样对手隐藏牌，并严格复验玩家可见观测不变。
- 公开出牌历史会约束样本保留近期已知牌，但当前 belief 仍是近似模型。
- `rollouts` 是初始样本数；分差低于 `confidence` 时按 `batch` 增加到 `max-rollouts`。
- `annotate=true` 让基础模型执行动作、搜索只记录教师目标，适合 DAgger 数据采集。
- `gate=N` 可按基础模型前两名 logit 分差跳过高置信局面，仅用于成本诊断。
- `crn=true`（默认）让同一轮不同候选共享随机数，降低候选比较方差。
- 执行搜索默认避免在相同公开状态重复同一动作，并原子完成固定数量选择与牌堆排序；
  可分别用 `avoid-repeats=false`、`auto-submit=false`、
  `avoid-choice-backtracking=false` 关闭；注释模式不会应用这些执行约束。
- 所有 `unsafe-rollout` 策略均为离线工具，推理服务会主动拒绝加载，不能直接上线。

当前验证过的低预算强度配置为：

```powershell
--policy-a "unsafe-rollout-cpu:models\structured-v2-search-dagger-v2.epoch-06.pt;candidates=3;rollouts=2;horizon=2;belief=true;exploration=0"
```

它在两批互不重叠、随机官方模组组合并交换座位的 400 局中对启发式得分 79.5%，
95% 区间为 75.60% 至 83.40%，没有循环恢复或强制兜底。该成绩属于离线混合策略，
不是 checkpoint 单独推理的成绩。

把记录式搜索标签编码成缓存，并按原始搜索分差过滤：

```powershell
.\.venv\Scripts\python.exe -m gtn_ai.build_recorded_teacher_cache `
  runs\search-dagger-adaptive-p0-20260814.jsonl.gz `
  runs\search-dagger-adaptive-p1-20260814.jsonl.gz `
  --config-checkpoint models\structured-v2-search-dagger-v2.epoch-06.pt `
  --min-teacher-margin 0.025 `
  --only-teacher-disagreements `
  --output datasets\structured-v2-search-adaptive-conf-v1
```

`--only-teacher-disagreements` 仅保留搜索教师推翻实际执行动作的局面，适合对稳定模型做
小步纠错；训练时仍应混入宽覆盖旧缓存，避免遗忘本来已经正确的动作。

蒸馏可混入旧缓存防止策略遗忘，或仅训练策略头：

```powershell
.\.venv\Scripts\python.exe -m gtn_ai.train_structured `
  datasets\structured-v2-search-adaptive-conf-v1 `
  --replay-cache datasets\structured-v2-search-dagger-v1 `
  --replay-ratio 2 `
  --trainable-scope all `
  --init-checkpoint models\structured-v2-search-dagger-v2.epoch-06.pt `
  --output models\structured-v2-search-continual-v1.pt
```

`--trainable-scope` 支持 `all`、`policy-heads` 和 `combat-policy-head`。多个兼容结构化
checkpoint 也可用 `structured-ensemble-cpu:A.pt|B.pt` 做低成本离线对照。

从结构化模型采集新鲜轨迹并做一次低学习率 PPO 更新：

```powershell
.\.venv\Scripts\python.exe -m gtn_ai.self_play `
  --games 2000 `
  --policy-0 structured-onpolicy-cpu:models\structured-v2-distilled.pt `
  --policy-1 structured-onpolicy-cpu:models\structured-v2-distilled.pt `
  --sample-mod-combinations `
  --record-decisions `
  --workers 4 `
  --output runs\structured-onpolicy-next.jsonl.gz

.\.venv\Scripts\python.exe -m gtn_ai.train_structured_actor_critic `
  runs\structured-onpolicy-next.jsonl.gz `
  --init-checkpoint models\structured-v2-distilled.pt `
  --output models\structured-v2-ppo-next.pt `
  --epochs 1 `
  --batch-size 128 `
  --shuffle-buffer 256 `
  --learning-rate 0.0000025 `
  --gae-lambda 0.95 `
  --device xpu
```

`self_play`、`arena`、结构化缓存、蒸馏和结构化 PPO 都默认每 10 秒输出一次低频进度。
交互终端显示单行进度条；重定向到日志时改为普通文本行，均包含吞吐、耗时和可计算时
的 ETA。热循环只做一次时间比较，性能影响可忽略；可用 `--progress-interval` 调整间隔，
或用 `--quiet` 完全关闭。结构化 PPO 的默认 `batch-size=128`、`shuffle-buffer=256` 是针
对本机内存实测后的安全值；不要在约 32 GB 内存的机器上直接恢复早期的 2048 洗牌缓冲。

`--device auto` 会依次选择 Intel XPU、CUDA、MPS 和 CPU。训练工程允许安装
PyTorch，生产游戏进程不依赖它；以后可让独立推理进程加载模型：

```powershell
.\.venv\Scripts\python.exe -m gtn_ai.inference_server `
  --policy neural:models\champion.pt `
  --host 127.0.0.1 `
  --port 8767
```

做座位互换的成对评估：

```powershell
python -m gtn_ai.arena `
  --pairs 500 `
  --policy-a linear:models\hashed-linear-v1.json `
  --policy-b heuristic `
  --sample-mod-combinations `
  --workers 8 `
  --output .runtime\arena-result.json
```

启动本机推理服务：

```powershell
python -m gtn_ai.inference_server `
  --policy linear:models\hashed-linear-v1.json `
  --host 127.0.0.1 `
  --port 8767
```

审计下载的旧回放：

```powershell
python -m gtn_ai.replay_audit "D:\QQ File\GTN-R-12728.gtnreplay"
```

## 目录

- `gtn_ai/environment.py`：生产引擎适配器和原子动作
- `gtn_ai/observation.py`：严格按玩家视角构造观测
- `gtn_ai/self_play.py`：并行自我对局和轨迹采集
- `gtn_ai/linear_model.py`：可执行的最小训练基线
- `gtn_ai/neural_model.py`：可变合法动作的神经策略/价值网络
- `gtn_ai/neural_training.py`：流式行为克隆与终局价值训练
- `gtn_ai/actor_critic_training.py`：严格 on-policy 的裁剪 PPO 策略/价值训练
- `gtn_ai/structured_features.py`：保留实体与区域关系的结构化 token 编码
- `gtn_ai/structured_model.py`：状态 Transformer 与合法动作交叉注意力策略/价值网络
- `gtn_ai/structured_cache.py`：v8 教师预测和结构化张量的版本化分片缓存
- `gtn_ai/structured_distillation.py`：软策略、教师最优动作与价值联合蒸馏
- `gtn_ai/belief_sampling.py`：公开观测等价的隐藏牌确定化与公开历史约束
- `gtn_ai/rollout_search.py`：离线有限深度、自适应 belief rollout 教师
- `gtn_ai/arena.py`：座位互换评估、置信区间和近似 Elo
- `gtn_ai/inference_server.py`：供线性或神经策略使用的本机推理 API
- `gtn_ai/replay_audit.py`：旧回放训练适用性审计
- `gtn_ai/diagnostics.py`：本地真人测试的决策、标记和私有快照记录
- `gtn_ai/diagnostic_data.py`：将合规 `.gtnai.zip` 转换为严格玩家决策数据
- `gtn_ai/diagnostic_relabel.py`：从可信私有快照生成高预算离线教师标签
- `gtn_ai/validate_loadouts.py`：遍历官方模组组合
- `docs/PROTOCOL.md`：观测、动作与信息边界
- `docs/TRAINING_ROADMAP.md`：从基线到强 AI 的路线
- `docs/ONLINE_INTEGRATION.md`：未来接入在线游戏的方案
- `docs/AI_1V1_TEST_GATE.md`：公开账号入口、服务器容量和诊断数据边界
- `docs/VALIDATION.md`：当前规则版本的实测覆盖与结果

## 当前边界

1. 当前线性模型用于验证数据、训练、评估和部署链路，不代表强度目标。
2. 旧回放通常没有保存每一步的完整合法动作集合，不能直接作为无偏行为克隆数据。
3. 这是一个部分可观测博弈。单个确定状态上的搜索不能等同于真正最优策略。
4. 官方规则或 AI 协议改变后必须重新生成轨迹并训练模型；版本与规则指纹会主动阻止
   误用旧模型。
5. 多人游戏的 1v1 页面已接入面向登录账号的隔离 AI 对局，但仍不进入正式匹配、统计、
   GR 或奖励。它使用生产引擎；线上 worker 只监听回环地址，默认串行推理并把脱敏诊断
   保存到独立持久目录。
6. 当前本机冠军为 `models/champion.pt`（actor-critic v8 GAE）；它在 2400 场全官方模组
   组合、双座位竞技中以 52.17% 得分率超过 v6，并显著超过启发式与历史 v4，但尚未
   接入游戏，也不应被称为理论最优策略。
7. 当前最强纯结构化候选仍为 `models/structured-v2-search-dagger-v2.epoch-06.pt`；早期
   独立 200 局为 71.0%，新一批同种子基线为 60.0%，说明纯模型尚未稳定达到 75%。以它
   为基础的低预算 belief 搜索在两批共 400 局中达到 79.5%。该混合策略已接入 Phelren
   真人入口，实测首次加载约 4.2 秒、首步搜索约 1.1 秒；当前只用于不计分的人机对局，
   不能用作正式匹配机器人。下一阶段优先用自动候选和真人标记的坏步重算搜索标签，再
   蒸馏回低延迟学生。

早期 `runs/final-validation.jsonl.gz` 与 `models/final-validation.json` 属于旧战斗协议，
已不能由当前协议加载；它们只曾用于确认链路，不是准备上线的强模型。

详见 [训练路线](docs/TRAINING_ROADMAP.md) 和
[在线接入](docs/ONLINE_INTEGRATION.md)。本次实测结果见
[验证记录](docs/VALIDATION.md)。
