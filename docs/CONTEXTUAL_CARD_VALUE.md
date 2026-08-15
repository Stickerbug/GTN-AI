# Contextual Card Value System

## Goal

Phelren evaluates a visible card instance, not only a card name. A `Light`
with `sprout` and `symbiosis`, a temporarily heavy `Light`, and an ordinary
`Light` are three different inputs.

The player-authored formula at <https://note.ms/gtnprinciples> is treated as a
useful human prior and vocabulary, not as a fixed reward function. Constants
such as a permanent M-to-E exchange rate or a universal draw multiplier are
not valid across every mod combination, deck, turn, and opponent.

## Representation

`gtn_ai.contextual_value` adds four layers of public information:

1. **Definition semantics**: base E/M costs, type, mod, base tags, response and
   trigger costs, event names, rule operations, and rule numeric operands.
2. **Instance semantics**: effective E/M costs, added and removed tags,
   temporary speed/heavy effects, fission, fusion, power, charge, bonus damage,
   and extra hits.
3. **Deck context**: separate summaries for hand, all visible owned zones,
   equipment, public opponent cards, and pregame picks/options.
4. **Resource context**: current and maximum H/E/M, hand pressure, active
   equipment trigger demand, and independent rule-operation supply signals.

No engine instance ID or hidden opponent card is exposed. Definition metadata
is loaded inside the AI process from the version-locked official ruleset.

## Dynamic valuation

The structured policy receives card, deck, opponent, resource, and action
tokens together. Transformer attention therefore learns a context-conditioned
function rather than a lookup table:

```text
action value = F(
  visible card instance,
  owned deck and active equipment,
  current H/E/M and statuses,
  public opponent state,
  enabled official mods,
  decision history
)
```

The weights inside `F` remain trainable. They can change when training data
contains a different mod combination, deck composition, matchup, or human
correction. `transition_value_components()` exposes H/E/M, armor, status,
equipment, and zone deltas separately for future auxiliary outcome training;
it deliberately does not collapse them into one hand-written score.

## Compatibility

`StructuredModelConfig.contextual_value_features` defaults to `False`.
Existing checkpoints therefore retain byte-for-byte input behavior. A new
cache enables the representation explicitly, and an old checkpoint can seed
the new model because tensor dimensions do not change.

Smoke pipeline:

```powershell
.\.venv\Scripts\python.exe -m gtn_ai.build_structured_cache `
  runs\trajectory.jsonl.gz `
  --teacher models\champion.pt `
  --output datasets\contextual-value-v1 `
  --contextual-value-features `
  --device xpu `
  --overwrite

.\.venv\Scripts\python.exe -m gtn_ai.train_structured `
  datasets\contextual-value-v1 `
  --init-checkpoint models\structured-v2-search-combined-m05-head-v1.pt `
  --output models\structured-contextual-value-v1.pt `
  --device xpu
```

Search-labelled combat caches should be trained after the broad cache. Human
corrections remain the highest-value source for rare interactions such as
bleed plus `Light` into `Plank`, harmful equipment targets, and resource plans
that require several turns.

## Conservative residual policy

Training the whole contextual model directly can erase useful behavior already
present in the champion. The supported deployment shape therefore keeps two
frozen representations:

- the original encoder and policy produce the unchanged base logits;
- the contextual encoder describes the same state with card-instance and deck
  context;
- a small correction head may promote only one of the base policy's top-k
  actions;
- zero initialization and a strict confidence gate make the initial policy
  exactly identical to the base policy.

Contextual caches store both encodings for every decision. This preserves exact
action alignment and lets old checkpoints remain usable. Low-margin search
disagreements are keep-base examples in every correction loss component; they
must not teach the ranking head to override the champion.

```powershell
.\.venv\Scripts\python.exe -m gtn_ai.train_correction `
  datasets\structured-contextual-search-paired-v1 `
  --base models\structured-v2-search-combined-m05-head-v1.pt `
  --context-checkpoint models\structured-v2-contextual-broad-v1.pt `
  --contextual-value-features `
  --output models\structured-v2-contextual-residual-v2.pt `
  --device xpu
```

## Current result

The `v2` residual was trained from 12,233 search-labelled decisions. On the
held-out split it changed 3.83% of decisions and improved the conservative
target agreement from 71.78% to 72.19%. In 80 paired games against the
heuristic it scored 59.375%, equal to the safe base policy on the same seeds.
In 80 direct paired games against that base it scored 51.875% (41 wins, 38
losses, 1 draw), with no loop recovery or forced fallback.

This is evidence that the representation and residual path work without a
measurable regression, but not evidence of a stronger production policy. The
checkpoint is retained as an experiment and is not promoted to Phelren. The
next useful dataset should contain independently held-out, high-budget search
relabels of real human-game states, especially multi-turn resource plans and
modified-card interactions. Delayed human markers should identify a short
review window rather than be treated as exact labels for one action.

The first end-to-end human hard-example pass now exists. Delayed markers select
the marked decision and the two preceding decisions; a larger offline belief
search then labels each state without assuming that any marked-window action is
wrong. Twenty-one compatible states from five sessions survived strict replay
validation. They were split by whole session into 17 training and 4 validation
examples, then encoded as separate paired contextual caches. Correction and
anchor examples carry explicit weights through cache serialization, training,
and threshold evaluation. This dataset validates the pipeline only; it is far
too small to justify fitting or promoting a new policy.

## Validation gates

A contextual model is not promoted merely because training loss falls. It must:

- load every official card definition without semantic extraction failures;
- never expose hidden cards or engine instance IDs;
- remain legal-action-only in pregame, normal play, choices, and responses;
- improve on held-out search labels and human correction cases;
- beat the current champion on independent seeds without regressing pregame;
- pass targeted resource-synergy and modified-card-instance scenarios.
