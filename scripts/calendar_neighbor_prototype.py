"""Offline bounded calendar polishing; no production defaults or weight changes.

Bounded passes around the normal fully scored unconstrained winner and (when distinct)
the near-term-stable winner. Adds/removes anywhere; moves by one day.
Independent and prior-stable search lanes remain separate. Local termination
is reported separately from budget exhaustion; global optimality is not claimed.
All hooks are process-local and restored on failure.
"""

from contextlib import contextmanager
from inspect import signature
from itertools import combinations
from time import perf_counter

from run_analysis import weekly_schedule as planner
from run_analysis.aerobic_acceptance import assess_aerobic_exposure, AerobicAcceptance


class NoAcceptableCalendarError(RuntimeError):
    def __init__(self, as_of, message):
        super().__init__(message)
        self.as_of = as_of


from run_analysis.adaptive.calendar_operators import (
    calendar_neighbors, opening_opportunity_neighbors, paired_move_neighbors,
)


@contextmanager
def calendar_neighbor_policy(*, enabled=True, audit_opening=False, max_passes=1, enforce_acceptance=False):
    if max_passes < 1:
        raise ValueError("max_passes must be positive")
    original = planner.adaptive_run_day_offsets
    call_signature = signature(original)
    records = []

    def adaptive(*args, **kwargs):
        bound = call_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = bound.arguments
        diagnostics = inputs.get("_planner_diagnostics") or planner._PlannerSearchDiagnostics()
        inputs["_planner_diagnostics"] = diagnostics
        started = perf_counter()
        selected = original(**inputs)
        base_seconds = perf_counter() - started
        if not selected or inputs["request"].health_status != planner.CurrentHealthStatus.NORMAL:
            records.append({"base_seconds": base_seconds, "skipped": "empty_or_non_normal_health"})
            return selected
        independent = diagnostics.unconstrained_winning_candidate
        stable = diagnostics.near_term_stable_candidate
        if independent is None:
            raise ValueError("Prototype requires the normal full-score diagnostics")
        started = perf_counter()
        states = inputs["daily_states"]
        allowed = (set(range(len(states))) - set(inputs["forced_rest_offsets"] or ())
                   - set(inputs["completed_run_offsets"] or ()))
        limit = planner.NEAR_TERM_STABILITY_DECISION_OFFSETS
        has_prior = bool(inputs["prior_run_offsets"])
        prior_dates = {x for x in inputs["prior_run_offsets"] or () if x in allowed and x < limit}
        roles = {i: role for i, role in (inputs["prior_roles_by_offset"] or {}).items() if i in prior_dates}
        ordinary = sum(planner.typical_easy_distance(states[0])) / 2
        common = {key: inputs[key] for key in (
            "daily_states", "daily_state_options", "request", "config", "target_distance_range",
            "completed_miles_by_offset")}
        common["daily_state_options"] = common["daily_state_options"] or [[s] for s in states]
        # Invocation-local reuse only. Prefix materialization remains uncached
        # during reconciliation, as in the existing final search pass.
        common.update(recommendation_load_cache={}, projected_state_cache={})
        cache = {}
        evaluated = []
        cache_hits = 0
        assessments = {}
        unrepresentable = set()
        assessment_seconds = 0.0
        fact_seconds = 0.0

        def score(offsets, overrides=None):
            nonlocal cache_hits, assessment_seconds, fact_seconds
            key = (offsets, tuple(sorted((overrides or {}).items())))
            if key not in cache:
                details = {}
                joint = planner._joint_candidate_program_cost(
                    offsets, **common, expected_target_projector=inputs["expected_target_projector"],
                    prior_distance_midpoints_by_date=inputs["prior_distance_midpoints_by_date"],
                    role_overrides_by_offset=overrides, diagnostic_details=details,
                    reconcile_allocated_prefix=True, _collect_acceptance_facts=enforce_acceptance)
                coaching = float(details["finalized_coaching_cost"]) if "finalized_coaching_cost" in details else planner._adaptive_candidate_cost(
                    offsets, **common, role_overrides_by_offset=overrides,
                    include_mileage_path=False, include_recovery_interactions=False,
                    include_cadence_pressure=False)
                cache[key] = planner._score_breakdown(offsets, joint, coaching, ordinary, details)
                row = {"roles": overrides or {}, **cache[key].model_dump(mode="json")}
                if details.get('unrepresentable_distance_target'):
                    unrepresentable.add(key)
                    row['unrepresentable_distance_target'] = details['unrepresentable_distance_target']
                if enforce_acceptance:
                    fact_seconds += details.get("acceptance_fact_seconds", 0.0)
                    facts = details.get("aerobic_acceptance_facts")
                    if facts is None and key not in unrepresentable:
                        raise NoAcceptableCalendarError(states[0].as_of, "Finalized acceptance evidence is unavailable")
                    stamp = perf_counter()
                    assessment = (AerobicAcceptance(True, 'unrepresentable_distance_target', {})
                        if key in unrepresentable else assess_aerobic_exposure(**facts,
                        planning_mode=inputs.get("_acceptance_planning_mode")))
                    assessment_seconds += perf_counter() - stamp
                    if assessment.reason == "missing_context":
                        raise NoAcceptableCalendarError(states[0].as_of, "Candidate acceptance context is incomplete")
                    assessments[key] = assessment
                    row["acceptance"] = dict(rejected=assessment.rejected, reason=assessment.reason,
                                             evidence=assessment.evidence)
                evaluated.append(row)
            else:
                cache_hits += 1
            return cache[key]

        def eligible(row, overrides=None):
            key = (tuple(row.offsets), tuple(sorted((overrides or {}).items())))
            return key not in unrepresentable and (not enforce_acceptance or not assessments[key].rejected)

        if enforce_acceptance:
            # Normal search diagnostics don't carry finalized acceptance facts.
            # Re-evaluate only its two seeds, then cache them with every neighbor.
            independent = score(tuple(independent.offsets))
            stable = score(tuple(stable.offsets), roles) if stable is not None else None

        # Keep the existing candidate itself; extra options cannot worsen its
        # objective. Neighborhood seeds are fixed before scoring this pass.
        independent_seed = tuple(independent.offsets)
        stable_seeds = {tuple(selected)} if has_prior else set()
        best_independent, best_stable = independent, stable
        independent_search_seed = tuple(independent.offsets)
        rounds = []
        current_stable_seeds = stable_seeds
        stop_reason = "pass_limit"
        paired_escape_used = False
        paired_escape_evaluations = 0
        for pass_index in range(max_passes):
            pass_started = perf_counter()
            before_count, before_hits = len(evaluated), cache_hits
            before_independent = tuple(best_independent.offsets) if best_independent else independent_search_seed
            before_stable = tuple(best_stable.offsets) if best_stable is not None else None
            independent_neighbors = calendar_neighbors(before_independent, allowed)
            if planner._OPPORTUNITY_SEARCH_REPAIR.get():
                independent_neighbors = sorted(set(independent_neighbors) | set(
                    opening_opportunity_neighbors(before_independent, allowed)))
            independent_rows = ([best_independent] if best_independent else []) + [score(x) for x in independent_neighbors]
            best_independent = min((x for x in independent_rows if eligible(x)),
                key=lambda x: (x.total_cost, len(x.offsets), x.offsets), default=None)
            stable_rows = [best_stable] if best_stable is not None else []
            stable_neighbors = {x for seed in current_stable_seeds for x in calendar_neighbors(seed, allowed)}
            if planner._OPPORTUNITY_SEARCH_REPAIR.get():
                stable_neighbors.update(x for seed in current_stable_seeds
                                        for x in opening_opportunity_neighbors(seed, allowed))
            # Preserve the existing protected-date and retained-purpose lane.
            for offsets in sorted(stable_neighbors | set(independent_neighbors)):
                if has_prior and {x for x in offsets if x < limit} == prior_dates:
                    stable_rows.append(score(offsets, roles))
            best_stable = min((x for x in stable_rows if eligible(x, roles)),
                             key=lambda x: (x.total_cost, x.offsets), default=None)
            # Check one bounded escape neighborhood at a stalled winner. Two
            # coordinated moves need not win as separate intermediate plans.
            if (planner._OPPORTUNITY_SEARCH_REPAIR.get() and not paired_escape_used
                    and best_independent is not None
                    and tuple(best_independent.offsets) == before_independent):
                paired_escape_used = True
                escape_before = len(evaluated)
                escapes = paired_move_neighbors(before_independent, allowed)
                independent_rows = [best_independent, *[score(x) for x in escapes]]
                best_independent = min((x for x in independent_rows if eligible(x)),
                    key=lambda x: (x.total_cost, len(x.offsets), x.offsets), default=None)
                stable_escapes = set(escapes)
                if best_stable is not None:
                    stable_escapes.update(paired_move_neighbors(tuple(best_stable.offsets), allowed))
                stable_rows = [best_stable] if best_stable is not None else []
                for offsets in sorted(stable_escapes):
                    if has_prior and {x for x in offsets if x < limit} == prior_dates:
                        stable_rows.append(score(offsets, roles))
                best_stable = min((x for x in stable_rows if eligible(x, roles)),
                    key=lambda x: (x.total_cost, x.offsets), default=None)
                paired_escape_evaluations = len(evaluated) - escape_before
            if best_independent is None and best_stable is None:
                raise NoAcceptableCalendarError(states[0].as_of,
                    "No acceptable calendar in the evaluated candidate pools; no invalid fallback selected")
            rounds.append({"pass": pass_index + 1, "seconds": perf_counter() - pass_started,
                           "new_evaluations": len(evaluated) - before_count,
                           "cache_hits": cache_hits - before_hits,
                           "independent_offsets": list(best_independent.offsets) if best_independent else None,
                           "independent_cost": best_independent.total_cost if best_independent else None,
                           "stable_offsets": list(best_stable.offsets) if best_stable else None,
                           "stable_cost": best_stable.total_cost if best_stable else None})
            after_stable = tuple(best_stable.offsets) if best_stable is not None else None
            stable_covered = after_stable is None or after_stable in current_stable_seeds
            after_independent = tuple(best_independent.offsets) if best_independent else before_independent
            if (after_independent == before_independent
                    and after_stable == before_stable and stable_covered):
                stop_reason = "unchanged_neighborhood_winners"
                break
            current_stable_seeds = {after_stable} if after_stable is not None else current_stable_seeds
            independent_search_seed = after_independent
        tolerance = planner._near_term_stability_tolerance(best_independent.total_cost if best_independent else 0)
        winner = best_independent or best_stable
        if (best_stable is not None and (enforce_acceptance or not planner._has_underfunded_aerobic_exposure_gap(
                tuple(best_stable.offsets), planned_midpoint_miles=best_stable.planned_midpoint_miles,
                integrated_target_low_miles=best_stable.integrated_target_low_miles,
                ordinary_easy_midpoint=ordinary, target_distance_range=inputs["target_distance_range"]))
                and (best_independent is None or best_stable.total_cost <= best_independent.total_cost + tolerance)):
            winner = best_stable
        changed_lane = winner is best_stable and (best_independent is None or winner.offsets != best_independent.offsets)
        polish_seconds = perf_counter() - started
        selection_evaluated_count = len(evaluated)
        opening_audit = None
        if audit_opening and not records:
            # Diagnostic only: do not feed a second neighborhood into selection.
            audit_started = perf_counter()
            alternatives = [score(x, roles if changed_lane else None)
                            for x in calendar_neighbors(tuple(winner.offsets), allowed)]
            better = sorted((x for x in alternatives if eligible(x, roles if changed_lane else None)
                             and x.total_cost < winner.total_cost - 1e-9),
                            key=lambda x: (x.total_cost, x.offsets))
            opening_audit = {"seconds": perf_counter() - audit_started,
                             "neighbor_count": len(alternatives), "improving_count": len(better),
                             "best": better[0].model_dump(mode="json") if better else None,
                             "affects_selection": False}
        records.append({"as_of": states[0].as_of.isoformat(), "base_seconds": base_seconds,
                        "neighbor_seconds": polish_seconds, "evaluated_count": selection_evaluated_count,
                        "opening_neighbor_audit": opening_audit,
                        "candidate_acceptance_enabled": enforce_acceptance,
                        "acceptance_seconds": assessment_seconds,
                        "acceptance_fact_seconds": fact_seconds,
                        "acceptance_checks": len(assessments),
                        "acceptance_rejected": sum(x.rejected for x in assessments.values()),
                        "original_selected": diagnostics.winning_candidate.model_dump(mode="json"),
                        "independent_seed": list(independent_seed), "stable_seeds": [list(x) for x in stable_seeds],
                        "winner": winner.model_dump(mode="json"),
                        "unconstrained": best_independent.model_dump(mode="json") if best_independent else None,
                        "stable": best_stable.model_dump(mode="json") if best_stable else None,
                        "evaluated": evaluated, "passes": len(rounds), "max_passes": max_passes,
                        "paired_escape_used": paired_escape_used,
                        "paired_escape_evaluations": paired_escape_evaluations,
                        "rounds": rounds, "stop_reason": stop_reason})
        diagnostics.unconstrained_winning_candidate = best_independent
        diagnostics.near_term_stable_candidate = best_stable
        diagnostics.winning_candidate = winner
        diagnostics.stability_tolerance = tolerance
        diagnostics.stability_tie_break_applied = changed_lane
        diagnostics.winning_role_overrides = dict(roles) if changed_lane else {}
        return list(winner.offsets)

    try:
        if enabled:
            planner.adaptive_run_day_offsets = adaptive
        yield records
    finally:
        planner.adaptive_run_day_offsets = original
