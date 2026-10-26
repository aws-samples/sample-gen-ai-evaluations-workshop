"""End-to-end driver for a full (turned-up) optimization run.

Runs the same loop the notebook shows, but headless and at production fan-out
(model_config.PRODUCTION_RUN_EXAMPLE), writing results into data/committed_run/ so a
real run can replace the illustrative committed fixtures.

    python scripts/run_optimization.py all        # baseline + hill climb, write fixtures
    python scripts/run_optimization.py baseline    # just the source-vs-target baseline
    python scripts/run_optimization.py climb        # just the hill climb on the target

This is a thin orchestrator over the module's own pieces:
  - gateway/setup_gateway.py must have provisioned the single Gateway first
  - agent/travel_agent.py is the workload
  - evaluators/custom_evaluators.py scores sessions (fixed judge)
  - optimizer.py proposes candidates; hillclimb.py runs the loop

The AgentCore Runtime deploy + fan-out invocation and the AgentCore evaluation job
submission are wired in the notebook (Sections 3-6); this script imports those same
helpers. Left as a thin, documented entry point so the expensive wiring lives in one
place rather than being duplicated here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import model_config as mc  # noqa: E402


def baseline() -> None:
    print(f"[baseline] source={mc.SOURCE_MODEL.gateway_model} "
          f"target={mc.TARGET_MODEL.gateway_model} judge={mc.JUDGE_MODEL.model_id}")
    print("[baseline] wire the Runtime deploy + fan-out from the notebook helpers, "
          "score with the fixed judge, write data/committed_run/baseline.json")
    raise SystemExit("Not wired for headless run yet — use the notebook, or wire the "
                     "shared helpers here once the Gateway/Runtime path is validated.")


def climb() -> None:
    cfg = mc.PRODUCTION_RUN_EXAMPLE
    print(f"[climb] {cfg.candidates_per_generation} candidates x "
          f"{cfg.sessions_per_candidate} sessions x {cfg.scenarios_per_session} scenarios, "
          f"up to {cfg.max_generations} generations, stop gain<{cfg.min_accuracy_gain_to_continue}")
    raise SystemExit("Not wired for headless run yet — use the notebook.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["all", "baseline", "climb"])
    args = parser.parse_args()
    if args.action in ("all", "baseline"):
        baseline()
    if args.action in ("all", "climb"):
        climb()


if __name__ == "__main__":
    main()
