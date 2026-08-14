# 观测与动作协议

## 原则

策略只能看到与正常玩家相同的信息。生产引擎仍然是唯一规则权威，AI 不自行计算
“应当合法”的动作；环境从引擎生成合法动作，策略只能从该列表中选择。

当前版本：

- `observation.schema_version = 3`
- `action.schema_version = 3`
- `trajectory.schema_version = 4`
- HTTP `api_version = 1`

## 观测

主要字段：

- `loadout.official_mods`：本局最终生效的官方模组
- `loadout.fingerprint`：本局模组组合指纹
- `loadout.ruleset_fingerprint`：全部官方内容与核心 1v1 代码的统一规则指纹
- `self`：自己的 H/E/M、状态、装备、手牌与允许查看的牌堆信息
- `opponent`：公开资源、状态、装备、牌堆数量和被展示的牌
- `pending`：当前反制、选择、排序等窗口的公开数据
- `public_history`：已经发生的公开动作

预对局时 `phase = "pregame"`，`self` 改为当前玩家的配装与选牌视图：自己的候选、
已选卡、下一牌类和刷新次数均可见。对手只公开是否已选倾向、选牌进度和是否完成；
双方都选完倾向后才展示倾向名称，对手的选牌内容始终隐藏。

引擎的 `instance_id` 不会进入协议。手牌、装备和候选项使用当前观测内的 `slot`。
槽位只在该次决策中有效，服务端执行前必须再次验证动作仍在合法列表中。

失明、展示手牌、护目镜等效果由生产引擎的公开状态和适配器共同投影。对手未知手牌
只提供数量；不会因训练便利泄漏真实牌名。

## 动作

常见动作：

- `select_opening_event` / `reroll_opening_event`：选择或刷新配装倾向
- `confirm_opening_reveal`：双方倾向揭示后开始自己的选牌
- `draft_pick` / `draft_reroll`：十五轮三选一或刷新候选
- `select_pregame_choice` / `toggle_pregame_choice`：倾向的单选或多选
- `append_pregame_order` / `reset_pregame_order`：花序编排的完整牌序
- `submit_pregame_choice`：提交倾向子选择
- `play_card`：`hand_slot`，以及引擎要求时的目标 `choice`
- `respond`：反制手牌槽位，`null` 表示放弃
- `use_trigger`：装备槽位和需要时的目标
- `select_choice`：选择一个候选槽位
- `toggle_choice`：多选中切换一个候选槽位
- `append_choice_order`：向排序结果追加候选槽位
- `submit_choice`：提交当前多选/排序
- `default_choice`：选择引擎提供的默认结果
- `resolve_choice`：目标、花色、模式、确认或取消等结构化选择
- `end_turn`

每个动作都应保持原子性。策略不直接传入引擎对象、卡牌实例或任意脚本参数。

合法动作列表是面向策略的语义动作空间。若一次出牌必然打开候选为空、且取消后不会
继续结算的选牌窗口，该出牌不会被列为合法动作；已经打开的可取消窗口仍会提供取消
动作。生产引擎给出的候选快照（包括空列表）始终优先于按当前牌区重建，且正在打出的
源牌不会成为自己的候选项。这可避免策略学习“打出、取消、再次打出”的无意义循环，
但不会改动正式游戏规则。

## 规则兼容

模型保存训练时的 `ruleset_fingerprint`。推理时如果观测中的指纹不同，模型抛出明确
错误。在线接入方应记录错误并回退到启发式策略，而不是绕过检查。

一个卡池组合的 `loadout.fingerprint` 可以不同；跨模组组合训练的模型仍共享同一个
全局规则指纹。这样模型可以覆盖全部组合，同时对规则更新保持敏感。

## 新回放训练快照

如果以后希望用真实玩家行为克隆，每个玩家决策应额外保存：

```json
{
  "ai_decision": {
    "observation": {"schema_version": 3},
    "legal_actions": [{"schema_version": 3, "kind": "...", "payload": {}}],
    "selected_action": {"schema_version": 3, "kind": "...", "payload": {}}
  }
}
```

应保存动作发生前的玩家视角，而不是动作后的全局状态。对隐私敏感的昵称、聊天、IP
和账号信息不属于训练观测。
