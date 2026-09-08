# Adaptive Time-aware Task Planning with LLMs for Efficient Execution under Uncertainty

**Appendix:** [IEEE-RAL-Appendix](IEEE-RAL-Appendix)

This repository contains an AI2-THOR task scheduling framework for household tasks with temporal constraints. The main planner updates task-duration beliefs during execution and uses monitoring actions to reduce schedule risk. The repository also includes baseline planners, task-generation utilities, AI2-THOR scene assets, and optional offline analysis scripts.

## Repository Layout

- `src/dag_bayesian.py`: Main entry point for the adaptive DAG-Bayesian planner.
- `src/core/`: Scheduler, agent belief state, monitoring logic, Bayesian updates, and particle-filter updates.
- `src/scheduler/`: Primitive-action handling, temporal-constraint handling, and scheduler heuristics.
- `src/simulation/`: AI2-THOR controller setup and simulator execution support.
- `src/baselines/`: Comparison baselines, including CPM, EDF, CAP, and ProgPrompt.
- `src/models/`: Shared task, subtask, scheduler-state, and simulation-node data structures.
- `src/utils/`: Configuration, logging, task parsing, task generation, task validation, and visualization helpers.
- `assets/tasks/`: Structured task JSON files used by the planner and batch runner.
- `assets/prompts/`: Prompt templates used by the task-generation pipeline.
- `assets/scene_knowledge/`: Scene object metadata and AI2-THOR environment descriptions.
- `assets/cache/ai2thor_nav_graphs/`: Cached navigation graphs for selected AI2-THOR scenes.
- `scripts/run_all_ai2thor_batch.py`: YAML-driven AI2-THOR batch runner.
- `scripts/data_gen/`: Utilities for generating and converting task datasets.
- `scripts/offline/`: Optional planner-level experiment utilities that avoid full simulator execution.
- `assets/result_analysis/`: Optional analysis and table-generation utilities for experiment outputs.

## Environment

Create the project environment from `environment.yaml`, then activate it:

```bash
conda env create -f environment.yaml
conda activate adaptive-time-aware-task-planner
```

LLM-based components require an API key:

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

For headless AI2-THOR execution, use either the runner's `--cloud-rendering` option or an external display backend such as Xvfb.

## Primary Workflows

### Run the Adaptive Planner

The main planner runs a selected task in an AI2-THOR scene. If no task is supplied, the script uses its default case, instruction, and scene arguments.

```bash
python src/dag_bayesian.py \
  --scene FloorPlan13 \
  --case tasks_3_constraints_2 \
  --instruction 08_heat_the_potato_using_microwave_and_make_a_coffee_and_wash_all_fork_and_spoon.json \
  --cloud-rendering
```

Useful arguments include:

- `--scene`: AI2-THOR scene name, such as `FloorPlan1`, `FloorPlan13`, or `FloorPlan422`.
- `--case`: Task-set folder name under `assets/tasks/<task_folder>/`.
- `--instruction`: Instruction JSON filename or instruction selector.
- `--init_prior_mean`: Initial task-duration mean used by the Bayesian planner.
- `--belief-update-method`: `bayesian` or `particle_filter`.
- `--cloud-rendering`: Use AI2-THOR cloud rendering in headless environments.

### Run Baselines

CPM and EDF use the same structured task assets as the adaptive planner:

```bash
python src/baselines/cpm.py --scene FloorPlan13 --case tasks_3_constraints_2 --simulation --cloud-rendering
python src/baselines/edf/dag_edf.py --scene FloorPlan13 --case tasks_3_constraints_2 --simulation --cloud-rendering
```

The LLM-based baselines take a natural-language instruction:

```bash
python src/baselines/cap/cap_ai2thor.py \
  --scene FloorPlan13 \
  --instruction "make a coffee and heat the bread using the microwave" \
  --cloud-rendering

python src/baselines/progprompt/prog_ai2thor.py \
  --scene FloorPlan13 \
  --instruction "make a coffee and heat the bread using the microwave" \
  --cloud-rendering
```

### Run an AI2-THOR Batch

Use `scripts/run_all_config.yaml` to configure scenes, task folders, planner scripts, ablation settings, and retry behavior. The batch runner resolves config paths relative to `scripts/`.

```bash
python scripts/run_all_ai2thor_batch.py --config run_all_config.yaml --dry-run
python scripts/run_all_ai2thor_batch.py --config run_all_config.yaml
```

Important config fields:

- `scene_type` and `scene_lists`: Select AI2-THOR scenes.
- `approaches`: Scheduler script paths for non-LLM planners.
- `llm_scripts`: LLM baseline script paths.
- `task_folder_name`: Task folder under `assets/tasks/`.
- `ablation_configs`: Planner variants to sweep.
- `init_prior_configs`: Initial duration-prior settings.

### Generate Structured Tasks

Task-generation utilities live under `scripts/data_gen/`. To convert generated natural-language instructions into structured task JSON files:

```bash
python scripts/data_gen/nl_to_json.py \
  --task-counts 2 3 \
  --constraint-counts 1 2 \
  --version v1 \
  --prompt-file e2e_generator_ver12_kitchen.txt
```

The tracked prompt templates are:

- `assets/prompts/e2e_generator_ver12_kitchen.txt`
- `assets/prompts/e2e_generator_ver13_real.txt`

## Optional Offline Utilities

The offline scripts are included for faster planner-level checks and large parameter sweeps. They are not the primary simulator entry point; use them when you want to compare schedules without executing the full AI2-THOR action loop.

### Single Offline Run

```bash
python scripts/offline/offline_experiment.py \
  --approach bayesian \
  --ablation-config DEFAULT \
  --init-prior-config CORRECT_ESTIMATE \
  --task-folder-name sampled_10_instruction_set_for_final_experiment_set_b \
  --case tasks_3_constraints_2 \
  --scene FloorPlan13 \
  --instruction 01_boil_potato_and_heat_the_bread_using_microwave_and_put_apple_and_lettuce_in_fridge.json \
  --beam-bound 10,10 \
  --belief-update-method bayesian \
  --gt-distribution constant \
  --gt-seed 42 \
  --eta 0.1 \
  --nav-graph-source ai2thor_controller \
  --output-path assets/results/offline_single_run.json
```

Supported offline approaches are `bayesian`, `edf`, and `cpm`.

### Offline Suite Runner

```bash
python scripts/offline/run_experiment_suite.py --suite scalability --dry-run
python scripts/offline/run_experiment_suite.py --suite eta_sensitivity
python scripts/offline/run_experiment_suite.py --suite monitoring_budget
python scripts/offline/run_experiment_suite.py --suite pf_vs_bayesian
```

Tracked offline config files:

- `scripts/offline/config/batch_config.yaml`
- `scripts/offline/config/oracle_reference_config.yaml`
- `scripts/offline/config/scalability_config.yaml`
- `scripts/offline/config/eta_sensitivity_config.yaml`
- `scripts/offline/config/monitoring_budget_config.yaml`
- `scripts/offline/config/pf_vs_bayesian_gaussian_bayesian_config.yaml`
- `scripts/offline/config/pf_vs_bayesian_gaussian_particle_filter_config.yaml`
- `scripts/offline/config/pf_vs_bayesian_lognormal_bayesian_config.yaml`
- `scripts/offline/config/pf_vs_bayesian_lognormal_particle_filter_config.yaml`
- `scripts/offline/config/pf_vs_bayesian_mixture_bayesian_config.yaml`
- `scripts/offline/config/pf_vs_bayesian_mixture_particle_filter_config.yaml`

### Result Analysis

Analysis utilities under `assets/result_analysis/` operate on generated result directories. For example:

```bash
python -m assets.result_analysis.offline_comparison \
  --base_dir assets/results \
  --batch_dirname offline_exp_result/offline_batch_pf_vs_bayesian \
  --oracle_dirname offline_oracle_reference \
  --task_folder sampled_10_instruction_set_for_final_experiment_set_b \
  --tolerance-sweep 5.0 8.0 12.5 15.0
```

## Generated Outputs

The scripts create result and log files at runtime. Common generated locations include:

- `assets/results/`: Planner outputs, offline reports, and analysis summaries.
- `logs/`: Batch-run logs and per-task execution logs.
- `assets/runtime_state/dynamic/`: Runtime object-state snapshots.

These directories are treated as generated artifacts rather than hand-maintained source files.

## Notes

- AI2-THOR simulator runs may take several minutes per task, especially when LLM baselines are enabled.
- The cached navigation graphs in `assets/cache/ai2thor_nav_graphs/` reduce repeated AI2-THOR initialization overhead for supported scenes.
- Offline makespan values are planner-level estimates. Full AI2-THOR simulation timings can differ because they depend on executed primitive actions and simulator state.
