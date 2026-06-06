# Day 3 Report — Latent Distance as a Planning Signal

**Date:** June 2026  
**Evaluation Test:** Horizon 2 — Operational Evaluation  
**Metric:** RQ1 — Does latent distance meaningfully track physical progress toward the goal?

---

## Goal

The goal for Day 3 was to instrument the LeWM MPC planning loop and produce the first empirical measurement of the planner's core metric: **latent distance to goal at each replanning step.**

This directly tests **Assumption 3** from the project's implicit assumption inventory:

> *Latent distance is a meaningful proxy for physical progress — points close in Z correspond to physically similar states, and decreasing latent distance to the goal reflects genuine task progress.*

If Assumption 3 holds, latent distance should decrease monotonically toward zero as the robot approaches the goal, and episodes with lower terminal latent distance should succeed more often than those with higher terminal latent distance.

---

## Strategy and Instrumentation

LeWM's planner operates inside the `CEMSolver` class in the `stable-worldmodel` library. At each MPC replanning step, `CEMSolver.solve()` is called with an `info_dict` containing the current environment observations. This is the natural interception point.

We subclassed `CEMSolver` with an `InstrumentedCEMSolver` that overrides `solve()` to log measurements before delegating to the parent implementation. Two measurements are taken per environment per CEM call:

**Latent distance** — the L2 norm between the current state embedding and the goal embedding, both computed using LeWM's own `encode()` method (applying the full encoder + projector pipeline, i.e. the same embedding the planner optimises internally):

```python
enc_current = self.model.encode({'pixels': px})
enc_goal    = self.model.encode({'pixels': gl})
lat_dist = torch.norm(
    enc_current['emb'].squeeze() - enc_goal['emb'].squeeze()
).item()
```

**Physical distance** — the L2 norm between the current environment state vector and the goal state vector, extracted directly from the Push-T simulator's `state` and `goal_state` observations. This serves as ground truth for actual task progress.

Evaluation was run on 50 fixed episodes sampled from the Push-T expert dataset, using the paper's exact protocol: `horizon=5`, `receding_horizon=5`, `action_block=5`, `goal_offset=25`, `eval_budget=500`. Environment state is initialised from the dataset via callables (`_set_state`, `_set_goal_state`) to ensure physically meaningful starting conditions.

---

## Results

### Overall performance

Across 50 fixed episodes: **mean success rate ~70%** (range 60–80% across runs due to GPU non-determinism — see finding 3 below).

### Finding 1 — Latent distance correlates weakly with physical distance

**Pearson r = 0.18** between latent distance and physical distance across all logged steps.

This is statistically above zero but practically very weak. The latent embedding captures some physical structure — as expected from a model trained on expert trajectories — but is far too noisy to serve as a reliable planning metric. Assumption 3 is **violated**.

### Finding 2 — Latent distance trajectories are indistinguishable between success and failure

Both successful and failed episodes show decreasing latent distance over the episode at approximately the same rate. The mean curves are nearly identical and the variance bands completely overlap. You cannot predict episode outcome from the latent distance trajectory.

Physical distance trajectories, by contrast, clearly separate the two groups: successful episodes start with lower physical distance (~220 units) and maintain it, while failed episodes start higher (~300 units) with greater variance.

**The implication:** success is largely determined by initial physical conditions, not by planning behaviour in latent space. The planner executes a similar latent-space strategy regardless of whether the episode will succeed or fail.

### Finding 3 — GPU non-determinism violates Assumption 6

Repeated runs with identical episode sets and seeds produce different success rates (60–80% range). This is traceable to non-deterministic CUDA parallel reduction operations — the same mathematical computation produces slightly different floating-point results each run. This directly violates **Assumption 6** (dynamics are deterministic). All results are therefore reported as distributions rather than point estimates.

### Finding 4 — Warm-start state bleeds across episodes

The first evaluation run after solver initialisation is faster (~4 CEM calls to convergence) and more consistent than subsequent runs. This is caused by `WorldModelPolicy` carrying `_action_buffer` and `_next_init` state across `evaluate()` calls. The warm-start assumption (Assumption 19/20) introduces inter-episode dependency that invalidates the assumption of independent episode evaluation. Fix: recreate solver, policy, and world fresh before each evaluation.

### Finding 5 — Terminal latent distance does not predict success

Episodes that terminate with low latent distance do not succeed more reliably than those that terminate with high latent distance. The terminal latent distance distributions for success and failure overlap substantially. This is a second, independent violation of Assumption 3 — even at the end of an episode, being "close" in latent space does not correspond to task completion.

---

## Summary

| Assumption | Status | Evidence |
|---|---|---|
| A3: Latent distance ≈ physical progress | **Violated** | Pearson r = 0.18; trajectories indistinguishable by outcome |
| A6: Dynamics are deterministic | **Violated** | 60–80% success rate variance across identical runs |
| A19/20: Warm starting improves performance | **Partially violated** | Warm start bleeds across episodes, introducing dependency |

---

## Next Steps — Day 4

Day 4 will add prediction error logging — the gap between the predictor's imagined next latent state and the actual latent state observed after executing the action. This measures rollout drift directly and addresses RQ2.
