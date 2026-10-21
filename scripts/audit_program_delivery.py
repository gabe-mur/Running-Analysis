"""Synthetic allocation/cutoff controls, not searched or recovery-cleared plans."""
from datetime import date, timedelta
from time import perf_counter
import json

from run_analysis.program_delivery import ProgramExposure, TargetSegment, audit_program, compare_allocation, screen_development

ORIGIN = date(2026, 9, 25)


def regular(long=8.4, easy=4.4, quality=4.5, work=(300., 300., 300.)):
    return [(0, long, 'long', long, None, None),
            (2, easy, 'easy', easy, None, None),
            (4, quality, 'quality', 2., 'threshold', work)]


def ledger(motif, period=7):
    rows = []
    # All alternatives share exactly the same historical physical program.
    for low, high, pattern, cycle, source in ((-56, 0, regular(), 7, 'observed'),
                                              (0, 112, motif, period, 'planned')):
        for day in range(low, high):
            for i, (offset, miles, role, easy, purpose, bouts) in enumerate(pattern):
                if day % cycle == offset:
                    rows.append(ProgramExposure(f'{source}:{day}:{i}', ORIGIN+timedelta(days=day),
                        miles, role, source, easy, purpose, bouts))
    return rows


def panel():
    started = perf_counter()
    compressed = [(0, 8.8, 'long', 8.8, None, None),
                  (1, 4.9, 'easy', 4.9, None, None),
                  (2, 4.5, 'quality', 2., 'threshold', (300., 300., 300.))]
    frontloaded = [(base+offset, miles, role, easy, purpose, work)
                  for base in (0, 3, 6, 9) for offset, miles, role, easy, purpose, work in compressed]
    definitions = dict(
        hold=(regular(), 7, 17.3),
        extend_easy=(regular(easy=5.3), 7, 18.2),
        insert_support=(regular()+[(5, .9, 'easy', .9, None, None)], 7, 18.2),
        supported_long=(regular(long=8.8, easy=4.9), 7, 18.2),
        long_with_less_support=(regular(long=9.2, easy=4.2), 7, 18.2),
        quality_funded=(regular(quality=5.4, work=(300., 300., 300., 300.)), 7, 18.2),
        sparse=(regular(easy=3.)[:2], 7, 17.3),
        compressed=(compressed, 7, 18.2),
        frontloaded=(frontloaded, 28, 18.2),
        period_eight=(regular(long=9.2, easy=5.3, quality=5.3), 8, 17.3))
    kwargs = dict(coverage_start=ORIGIN-timedelta(days=56), coverage_end=ORIGIN+timedelta(days=112),
        evaluation_starts=[ORIGIN+timedelta(days=i) for i in range(28)], coverage_complete=True)
    reports, common = {}, {}
    for name, (motif, period, target) in definitions.items():
        exposures = ledger(motif, period)
        reports[name] = audit_program(exposures, [TargetSegment(kwargs['coverage_start'], kwargs['coverage_end'], target)], **kwargs)
        common[name] = audit_program(exposures, [TargetSegment(kwargs['coverage_start'], kwargs['coverage_end'], 18.2)], **kwargs)
    summaries = {}
    for name, report in reports.items():
        rows = report['intervals']
        first28 = next(r for r in rows if r['start'] == ORIGIN and r['days'] == 28)
        summaries[name] = dict(own_target_pattern=report['funding_evidence_status'],
            development_screen=screen_development(reports['hold'], report),
            common_growth_target_pattern=common[name]['funding_evidence_status'],
            target_ratio_range=report['target_ratio_range'],
            opening28={k: first28[k] for k in ('miles', 'target_miles', 'running_days', 'nonlong_miles',
                'easy_miles_lower', 'easy_miles_upper', 'total_quality_work_seconds', 'maximum_running_streak', 'largest_day_fraction')},
            opening28_max_gap=first28['gaps']['all']['maximum_bracketed_days'],
            opening28_vs_hold=next(r for r in compare_allocation(reports['hold'], report)
                                   if r['start'] == ORIGIN and r['end'] == ORIGIN+timedelta(days=28)))
    return dict(summaries=summaries, reports=reports, elapsed_seconds=perf_counter()-started,
        synthetic=True, searched_calendar=False, recovery_assessed=False, planner_influence=False,
        limitations=['Motifs are accounting controls, not recommended schedules',
            'Complete synthetic coverage, not an assumption that historical upload gaps were rest',
            'Easy-phase attribution is declared fixture intent, not measured physiological intensity',
            'No quality substitution, session-distance guard or dose-specific recovery clearance',
            'Finite shifted-window panel cannot establish optimality or daily-replan stability'])


if __name__ == '__main__':
    result = panel()
    result['intervals_per_program'] = len(result['reports']['hold']['intervals'])
    del result['reports']
    print(json.dumps(result, default=lambda x: x.isoformat() if isinstance(x, date) else str(x), indent=2))
