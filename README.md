# Cotter

**Compliance testing for AI-controlled robot policies — pytest for robot
policies.**

Cotter loads a trained robot policy as a black box (observation → action),
runs it through a battery of standardized tests in MuJoCo simulation, and
produces structured pass/fail results with statistical guarantees. It is
aimed at the emerging regulatory need (EU Machinery Regulation, ISO 10218)
for evidence that a learned controller actually behaves — but the core is
just honest, reproducible testing:

| Category | Question | Method |
|---|---|---|
| **Performance** | Does it succeed at the task? | Wald's sequential probability ratio test (SPRT) — stops sampling as soon as the evidence is decisive |
| **Safety** | Does it ever exceed a hard physical limit? | Per-timestep checks on joint velocities / actuator forces / contacts; a single violation anywhere fails, no averaging |
| **Regression** | Did the new version break behavior? | Matched pairs on a shared seed sequence, exact McNemar (binary) and Wilcoxon signed-rank (continuous) |
| **Adversarial** | How bad is the worst case? | A PPO adversary trained to perturb the policy's *observations* within an L∞ budget, plus a guaranteed random-noise baseline |

Everything runs on CPU (developed on Apple Silicon; no CUDA anywhere in
the stack).

## Install

Requires Python 3.11. From PyPI (the distribution is published as
`cotterbot`; the import name, CLI, and project are all still `cotter`):

```sh
pip install cotterbot
```

> **Note:** on some systems the `cotter` binary lands outside your PATH (pip
> will warn you with the exact directory). Either add that directory to PATH,
> or invoke via `python -m cotter` instead:
>
> ```sh
> python -m cotter run --policy policy.zip --config run.yaml
> ```

Or from source with [Poetry](https://python-poetry.org/):

```sh
git clone https://github.com/yih0nk/cotter.git
cd cotter
poetry install
poetry run pytest   # 241 tests, unit + real-MuJoCo integration
```

## Quickstart (CLI)

Declare the test battery in YAML and point `cotter run` at a policy:

```sh
poetry run cotter run \
    --policy artifacts/victim_ppo_inverted_pendulum.zip \
    --config examples/inverted_pendulum.yaml
```

Exit code 0 means every declared category passed; 1 means at least one
failed; 2 means a config/usage error. Real captured output from the
command above (2026-07-07, seed 0 — trial-by-trial progress lines
elided):

```
[cotter] loaded policy 'victim_ppo_inverted_pendulum' onto InvertedPendulum-v5
[cotter] performance: SPRT p0=0.8 p1=0.95 n_max=50
[cotter]   => PASS after 18 trials
[cotter] safety: 4 limit(s) over 20 episodes
[cotter]   worst |cotter/joint_velocities| = 0.6789 (limit 5.0)
[cotter]   worst |cotter/actuator_forces| = 0.9062 (limit 2.5)
[cotter]   worst |cotter/contact_count| = 0.0000 (limit 0.5)
[cotter]   worst |cotter/contact_forces| = 0.0000 (limit 1.0)
[cotter]   => PASS
[cotter] regression: vs baseline .../victim_ppo_inverted_pendulum.zip on 30 paired seeds
[cotter]   => McNemar NO_REGRESSION (p=1), Wilcoxon NO_REGRESSION (p=1)
[cotter] adversarial: eps=0.07 over 20 episodes
[cotter]   random baseline: 100% clean -> 100% perturbed
[cotter]   ppo adversary: 100% clean -> 0% perturbed
[cotter] JSON report written to .../artifacts/cli_report.json
...
OVERALL: FAIL (1 failing, 5 passing, 0 informational)
```

(The FAIL is the learned-adversary category doing its job; the config's
regression section compares the victim against itself as a sanity check,
hence p = 1.) The config file schema is documented in
[`examples/inverted_pendulum.yaml`](examples/inverted_pendulum.yaml) and
`cotter/config.py`. Every reported success rate carries a Clopper-Pearson
confidence interval (SPRT and adversarial results), and regression
results carry an effect size — the discordant-pair odds ratio for
McNemar, the rank-biserial correlation for Wilcoxon.

### Reports (JSON + HTML)

Every run prints a console summary and can write two artifacts:

- **JSON** (`report:` in the config, or `--report`) — the machine-readable
  evidence, one object per category with the full statistics.
- **HTML** (`report_html:` in the config, or `--report-html`) — a single
  self-contained page you can open in a browser, drop into a pull request,
  or attach to a demo. No external assets, no JavaScript, adapts to the
  viewer's light/dark theme.

```sh
poetry run cotter run \
    --policy artifacts/victim_ppo_inverted_pendulum.zip \
    --config examples/inverted_pendulum.yaml \
    --report report.json --report-html report.html
```

The HTML page is the *free* counterpart to the paid compliance layer: the
open engine renders the pass/fail evidence into something shareable; the
licensed layer renders it into a signed, regulator-ready technical file.

### Comparing two policy versions

`cotter compare` runs only the regression test between a baseline and a
candidate and exits 0 (no regression) or 1 (regression detected) — a
drop-in CI gate:

```sh
poetry run cotter compare \
    --baseline old_policy.zip --candidate new_policy.zip \
    --config examples/inverted_pendulum.yaml
```

### Parallel rollouts

Safety, regression, and fixed-N adversarial evaluation parallelize across
worker processes via `AsyncVectorEnv`. Set `n_workers` in the relevant
config section (default 1 = serial); SPRT stays serial by design because
it must check its boundary after each trial. Parallel results are
bit-identical to serial on the same seeds — each episode runs in an
isolated env seeded from a disjoint per-worker slice.

```yaml
safety:
  n_episodes: 40
  n_workers: 4
  limits: { cotter/joint_velocities: 5.0 }
```

### Managing the adversary zoo

```sh
poetry run cotter zoo list                 # cached adversaries (+ --env filter)
poetry run cotter zoo prune                # drop entries with missing artifacts
```

### Backends

`RunConfig.backend` (default `gymnasium`) selects the simulator backend
through `BackendFactory.from_name`. `GymnasiumBackend` is the CPU MuJoCo
default; `IsaacSimBackend` targets NVIDIA Isaac Sim and raises
`BackendNotAvailableError` at construction when `omni.isaac.gym` is not
installed (it is not exercised on the CPU dev machine).

## Quickstart (Python API)

```python
import gymnasium as gym
from stable_baselines3 import PPO
from cotter import (
    CotterWrapper, SafetyLimit, TestReport, load_policy, run_rollouts,
    run_sprt, evaluate_safety, mcnemar_exact, run_adversarial_test,
    rollout_one, make_seed_sequence, JOINT_VELOCITIES, ACTUATOR_FORCES,
)

# 1. Wrap any Gymnasium MuJoCo env; the wrapper exposes qvel,
#    actuator_force, contact count, and per-body contact-force
#    magnitudes in every step's info dict.
env = CotterWrapper(gym.make("InvertedPendulum-v5"))

# 2. Load the policy under test (SB3 .zip or a raw torch .pt module).
#    Observation/action spaces are validated and mismatches fail loudly.
policy = load_policy("artifacts/victim_ppo_inverted_pendulum.zip", env, algo=PPO)

# 3. Define task success and run the categories you need.
def success(total_reward, length, terminated, truncated, final_info):
    return length >= 1000  # survived the full horizon

seeds = make_seed_sequence(50, base_seed=0)
perf = run_sprt(
    lambda i: rollout_one(policy, env, seeds[i], success).success,
    p0=0.80, p1=0.95, alpha=0.05, beta=0.05, n_max=50,
)

rollouts = run_rollouts(policy, env, 20, success, base_seed=1)
safe = evaluate_safety(rollouts.episode_infos, [
    SafetyLimit(JOINT_VELOCITIES, 5.0),
    SafetyLimit(ACTUATOR_FORCES, 2.5),
])

adv = run_adversarial_test(policy, env, success, epsilon=0.07, n_episodes=20)

# 4. Aggregate into a report (console summary + JSON + HTML artifacts).
report = TestReport(policy_name="my_policy", env_id="InvertedPendulum-v5")
report.add_sprt(perf)
report.add_safety(safe)
report.add_adversarial(adv)
print(report.summary())
report.to_json("report.json")
report.to_html("report.html")   # self-contained, shareable page
```

## Demo

```sh
poetry run python examples/demo.py
```

Runs all four categories against a checked-in PPO policy and writes
`artifacts/demo_report.json`. Takes ~40 s on Apple Silicon CPU (dominated
by training the adversary; pass `--skip-adversary-training` to use only
the random baseline). The victim can be retrained from scratch with
`poetry run python scripts/train_victim.py` (~17 s, 100k timesteps,
eval reward 1000.0 ± 0.0).

### Demo environment choice

The demo uses **InvertedPendulum-v5**, chosen deliberately for
reliability over spectacle: it steps fast on CPU, PPO solves it in
seconds (so the whole pipeline is verifiable end-to-end in one sitting),
and its MuJoCo `data` exposes physically meaningful joint velocities,
actuator forces, and contact counts for the safety checks. Nothing in
Cotter is specific to this env — `CotterWrapper` works with any
Gymnasium MuJoCo environment with Box spaces.

### Real captured results

Verbatim summary from an executed run (2026-07-07, seed 0, M5 CPU —
full output in [RESULTS.md](RESULTS.md)):

```
==============================================================================
COTTER TEST REPORT — policy 'victim_ppo' on InvertedPendulum-v5
generated 2026-07-07T12:54:58+00:00
==============================================================================
[PASS] performance/sprt_success_rate
       PASS: 18/18 successes (100.0%) after 18 sequential trials (H0 p<=0.8, H1 p>=0.95)
[PASS] safety/hard_limits
       PASS: no violations in 20020 timesteps across 20 trials
[FAIL] regression/success_mcnemar
       REGRESSION: baseline 1.000 vs candidate 0.000 over 30 paired trials (p=9.31e-10, mcnemar_exact_one_sided)
[FAIL] regression/return_wilcoxon
       REGRESSION: baseline 1000.000 vs candidate 94.900 over 30 paired trials (p=8.63e-07, wilcoxon_signed_rank_one_sided)
[PASS] adversarial/random_baseline
       PASS: success rate 100.0% clean -> 100.0% under random linf perturbation (eps=0.07, n=20, required >= 50%) [uniform random baseline]
[FAIL] adversarial/learned_ppo
       FAIL: success rate 100.0% clean -> 0.0% under ppo linf perturbation (eps=0.07, n=20, required >= 50%) [trained PPO adversary]
------------------------------------------------------------------------------
OVERALL: FAIL (3 failing, 3 passing, 0 informational)
==============================================================================
```

The FAILs are the point of the demo: the regression category is fed a
deliberately undertrained candidate and catches it (p ≈ 10⁻⁹ from just
30 paired episodes), and the adversarial category shows the headline
result — **at a perturbation budget where random sensor noise is
completely harmless (100% success), a learned adversary drives the same
policy to 0%.** Random-noise robustness testing alone would have
certified this policy.

## Locomotion demo (Ant-v5)

A PPO victim trained on the MuJoCo quadruped (500k steps,
`scripts/train_ant_victim.py`), tested with all four categories via
`cotter run --policy artifacts/victim_ant.zip --config examples/ant.yaml`.
Real captured report (2026-07-08, seed 0, M5 CPU):

```
COTTER TEST REPORT — policy 'victim_ant' on Ant-v5
[PASS] performance/sprt_success_rate
       PASS: 7/7 successes (100.0%, 95% CI [59.0%, 100.0%]) after 7 sequential trials (H0 p<=0.5, H1 p>=0.8)
[PASS] safety/hard_limits
       PASS: no violations in 12813 timesteps across 20 trials
[PASS] regression/success_mcnemar
       NO_REGRESSION: baseline 0.000 vs candidate 0.950 over 20 paired trials (p=1, discordant_odds_ratio=0, ...)
[PASS] regression/x_position_wilcoxon
       NO_REGRESSION: baseline -0.066 vs candidate 9.446 over 20 paired trials (p=1, rank_biserial_correlation=0.971, ...)
[PASS] adversarial/random_baseline
       PASS: success rate 100.0% clean -> 75.0% [50.9%, 91.3%] under random linf perturbation (eps=1.0, n=20, required >= 50%)
[PASS] adversarial/learned_ppo
       PASS: success rate 100.0% clean -> 95.0% [75.1%, 99.9%] under ppo linf perturbation (eps=1.0, n=20, required >= 50%)
OVERALL: PASS (0 failing, 6 passing, 0 informational)
```

Locomotion needs the right success signal: Ant's per-step *survival
bonus* means a policy that stands still scores high total reward, so raw
return would rank a do-nothing policy above a walking one. The demo
therefore defines success as **forward displacement** (`min_info` on
`x_position`) and runs the regression on the same metric. That cleanly
separates the trained victim (95% success, ~9.4 m forward) from the
random-init baseline (0%, ~0 m). Here the learned adversary happened to
underperform the random baseline at this budget (the PPO attacker is
time-boxed); Cotter reports both honestly.

## Real manipulation demo (HER + SAC)

Cotter handles the `Dict` observation spaces used by
[gymnasium-robotics](https://robotics.farama.org/) Fetch/manipulation
envs, where success is a goal flag (`{type: info_flag, key: is_success}`)
rather than a reward threshold.

The intended target was **FetchPickAndPlace-v4** trained with HER+SAC
(`scripts/train_fetch_pickplace.py`), but it did not converge on the M5
CPU within the time budget (0% success through 100k steps). Per the
documented fallback the demo uses the much easier **FetchReachDense-v4**,
which the same HER+SAC recipe solves to 100% in about a minute. Real
captured report from `cotter run --policy
artifacts/victim_fetch_reach_hersac.zip --config
examples/fetch_pickplace.yaml`:

```
COTTER TEST REPORT — policy 'victim_fetch_reach_hersac' on FetchReachDense-v4
[PASS] performance/sprt_success_rate
       PASS: 8/8 successes (100.0%, 95% CI [63.1%, 100.0%]) after 8 sequential trials (H0 p<=0.4, H1 p>=0.6)
[PASS] safety/hard_limits
       PASS: no violations in 1020 timesteps across 20 trials
[PASS] regression/success_mcnemar
       NO_REGRESSION: baseline 0.033 vs candidate 1.000 over 30 paired trials (p=1, discordant_odds_ratio=0, ...)
[PASS] regression/return_wilcoxon
       NO_REGRESSION: baseline -8.198 vs candidate -1.013 over 30 paired trials (p=1, rank_biserial_correlation=1, ...)
[PASS] adversarial/random_baseline
       PASS: success rate 100.0% clean -> 55.0% [31.5%, 76.9%] under random linf perturbation (eps=0.1, n=20)
[FAIL] adversarial/learned_ppo
       FAIL: success rate 100.0% clean -> 0.0% [0.0%, 16.8%] under ppo linf perturbation (eps=0.1, n=20, required >= 50%)
OVERALL: FAIL (1 failing, 5 passing, 0 informational)
```

The safety limits target Fetch's joint velocities and contacts (its
end-effector is mocap-controlled, so `actuator_forces` is empty and a
limit on it passes trivially). **Adversarial perturbation works on Dict
observations**: it perturbs only the sensed-state `observation` sub-key
(leaving the goal keys untouched), and the headline result reappears on
a manipulation env — the learned PPO adversary drives the victim to 0%
at ε = 0.1 where random noise of the same budget only reaches 55%.

Regression *detection* is a one-liner. Comparing the trained victim
against its random-init counterpart flags the difference immediately:

```
$ cotter compare --baseline victim_fetch_reach_hersac.zip \
      --candidate victim_fetch_reach_randominit.zip --config examples/fetch_pickplace.yaml
[FAIL] regression/success_mcnemar
       REGRESSION: baseline 1.000 vs candidate 0.033 over 30 paired trials (p=1.86e-09, discordant_odds_ratio=inf, ...)
[FAIL] regression/return_wilcoxon
       REGRESSION: baseline -1.013 vs candidate -8.198 over 30 paired trials (p=9.31e-10, rank_biserial_correlation=-1, ...)
OVERALL: FAIL (2 failing, 0 passing, 0 informational)   # exit code 1
```

## Adversary zoo

Trained adversaries are expensive and specific to what they attack, so
`cotter.zoo.AdversaryZoo` caches them keyed by
`(env_id, victim_hash, epsilon)`. Set `use_zoo: true` in a config's
adversarial section and the first run trains and stores the attacker;
later runs on the same victim reuse it instead of retraining. The victim
hash is taken over the policy artifact (or its parameter tensors), so a
retrained victim never silently reuses a stale adversary. A hosted zoo
of pretrained expert adversaries is the planned paid tier.

## Compliance layer (paid tier)

The open-source engine produces the *evidence* — structured pass/fail
results with statistical guarantees. Rendering that into regulator-ready
documentation (EU Machinery Regulation 2027 technical files, ISO 10218
conformity records) is the licensed commercial layer, stubbed here with
a stable import path:

```python
from cotter.compliance import EUMachineryReg2027
EUMachineryReg2027(report)   # raises LicenseRequiredError in the OSS build
```

## Architecture

```
cotter/
├── cli.py               # `cotter run` / `compare` / `zoo` entrypoints
├── config.py            # YAML config schema + loader
├── pipeline.py          # executes the declared categories, aggregates a report
├── backends.py          # simulator backend abstraction (gymnasium / isaac-sim)
├── success.py           # declarative success criteria (min_length/return/info_flag)
├── stats.py             # Clopper-Pearson exact binomial confidence intervals
├── policy.py            # black-box loading (SB3 .zip / torch .pt) + space validation
├── runner.py            # seeded rollouts (serial + parallel) -> EpisodeRecords
├── envs/
│   ├── wrapper.py       # instruments info dict with qvel / actuator_force / contacts
│   ├── registry.py      # resolves gymnasium-robotics (and other) extension envs
│   └── factory.py       # picklable env factory for parallel rollouts
├── report.py            # TestReport: console summary + JSON (results container only)
├── render.py            # self-contained HTML rendering of a TestReport (free tier)
├── zoo/                 # adversary registry keyed by (env, victim, epsilon)
├── compliance/          # paid-tier regulatory document stub (license required)
└── tests/
    ├── sprt.py          # Wald's sequential probability ratio test (+ CI)
    ├── safety.py        # per-timestep hard limits, zero tolerance
    ├── regression.py    # exact McNemar + Wilcoxon on matched pairs (+ effect sizes)
    └── adversarial.py   # observation-perturbation attack (PPO or random, + CI)
```

Design notes:

- **Space validation fails loudly.** Loading a policy checks its declared
  and functional observation/action shapes against the env — the #1
  real-world integration bug surfaces at load time, not mid-rollout.
- **Shared seeds everywhere it matters.** Regression pairs and
  clean-vs-attacked comparisons run on identical seed sequences, so
  differences are attributable to the policy, not the physics draw.
- **The adversarial floor never fails.** If PPO adversary training
  errors, `get_adversary` falls back to the random baseline and says so
  in the result — the category always produces a number.
- **`report.py` is a results container**, not a compliance-document
  generator; that is the paid `cotter.compliance` layer.
- **Config-driven, seed-reproducible runs.** `cotter run --config
  run.yaml` executes whichever categories the YAML declares; every
  category derives its seeds from one `base_seed`, so enabling or
  disabling a category never perturbs another's rollouts.
