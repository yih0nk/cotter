# Changelog

## 0.3.0 (unreleased)

- **ONNX policy loader.** Load `.onnx` policies via `load_policy` (the new
  `OnnxPolicy`, backed by `onnxruntime`). Install the optional extra with
  `pip install cotterbot[onnx]`. Handles the common single-input
  Box-observation export shape, adding/stripping a batch axis
  automatically; multi-input (Dict-observation) models raise a clear error.
- **`cotter verify` command.** Check a report's integrity —
  `cotter verify report.json` recomputes the `content_sha256` and reports
  whether the report was modified; `--policy <path>` also re-hashes the
  policy against the manifest's `policy_sha256`. Exit 0 = verified, 1 =
  a check failed, 2 = missing/invalid report.
- **JUnit XML report output.** `report_junit:` in the config or
  `--report-junit` on `run`/`compare` writes a JUnit XML report (each
  category a `<testcase>`), so a Cotter run renders as native test results
  in GitHub Actions, GitLab, Jenkins, and other CI systems.
- **Markdown report output.** `report_md:` in the config or `--report-md`
  on `run`/`compare` writes a GitHub-flavored Markdown report — a
  shareable plain-text summary for PRs, issues, and wikis.
- **Action-space adversary.** A second attack surface: bounded
  perturbation of the victim's *action* (actuator noise / control-channel
  corruption), alongside the existing observation attack. Select via
  `adversarial.attack: observation | action | both`. Adds
  `run_action_adversarial_test`, `train_action_adversary`,
  `ActionPerturbationEnv`, and the action adversaries to the public API.
- **Pretrained adversary zoo + transfer attacks.** `cotter.zoo.PretrainedZoo`
  registers adversaries by **robot class** (not victim), and
  `transfer_attack` applies one to any compatible victim it never trained
  on. Enable in a config with `adversarial.pretrained: <robot-class>`;
  inspect with `cotter pretrained list` / `prune`. This is the open
  mechanism behind the pretrained-expert-per-robot-class tier.

## 0.2.0

**Reports**

- Free, self-contained **HTML report** — `cotter run --report-html` /
  `report_html:` in the config, or `TestReport.to_html()`. No external
  assets, no JavaScript, adapts to the viewer's light/dark theme.
- `cotter compare` now accepts `--report` (JSON), symmetric with
  `--report-html`, so a comparison can emit machine-readable evidence.

**Reproducibility (report schema v2)**

- Every report carries a **reproducibility manifest**: cotter version,
  dependency + platform versions, env id, base seed, and the policy
  artifact's sha256.
- Reports include **`content_sha256`**, a tamper-evident digest over the
  report body (excluding the timestamp and the digest itself) — a stable
  id that can be independently recomputed and verified.
- `cotter_report_version` bumped `1` → `2`. The added fields are
  additive; consumers that read the version tolerate both.

## 0.1.0

Initial PyPI release (`cotterbot`): the test engine (performance / safety
/ regression / adversarial), `cotter run` / `compare` / `list-envs` /
`zoo` CLI, JSON report, adversary zoo, and parallel rollouts.
