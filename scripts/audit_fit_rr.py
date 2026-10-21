"""Read-only FIT/ZIP inventory and RR screening; not a clinical HRV analysis.

No live import, repair, resampling, or DFA threshold interpretation. RR message
placement uses surrounding record timestamps only as approximate bounds.
"""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from pathlib import Path
import io
import json

import fitdecode
import numpy as np

from run_analysis.session_feedback import feedback_from_session_fields
from run_analysis.exercise_hrv import screen_rr
from run_analysis.web.upload_archive import fit_members


def rr_context_windows(batches, following, states, events, start, elapsed, window_seconds=120):
    """Conservative message-context diagnostics, not validated beat timestamps.

    Keep raw intervals unchanged. A source bracket is not a measured beat time;
    windows touching pauses or the partial end must not masquerade as continuous
    exercise. Timer state reflects message order and is reported as such.
    """
    if (len(batches) != len(following) or len(batches) != len(states)
            or not np.isfinite(elapsed) or elapsed <= 0
            or type(window_seconds) is not int or window_seconds <= 0):
        raise ValueError('Matching source contexts and positive duration required')
    values = np.array([x for _,rr in batches for x in rr],dtype=float)
    flags = screen_rr(values) if len(values) else np.zeros(0,dtype=bool)
    lower = np.array([(at-start).total_seconds() if at else np.nan for at,rr in batches for _ in rr])
    upper = np.array([(at-start).total_seconds() if at else np.nan for at,(_,rr) in zip(following,batches) for _ in rr])
    timer = np.array([state for state,(_,rr) in zip(states,batches) for _ in rr])
    pauses, stopped_at = [], None
    for event in sorted(events,key=lambda e:e['timestamp']):
        at = (event['timestamp']-start).total_seconds()
        if event['event_type'] in {'stop','stop_all','stop_disable','stop_disable_all'} and stopped_at is None:
            stopped_at = at
        elif event['event_type'] == 'start' and stopped_at is not None:
            pauses.append((max(0.,stopped_at),min(elapsed,at)))
            stopped_at = None
    if stopped_at is not None:
        pauses.append((max(0.,stopped_at),elapsed))
    windows = []
    for offset in range(0,int(np.ceil(elapsed)),window_seconds):
        end = min(offset+window_seconds,elapsed)
        selected = np.isfinite(lower)&np.isfinite(upper)&(lower>=offset)&(upper<=end)&(upper>=lower)&(timer=='running')
        overlap = sum(max(0.,min(end,b)-max(offset,a)) for a,b in pauses)
        windows.append(dict(approx_start_s=offset,approx_end_s=end,
            full_wall_time_window=end-offset==window_seconds,pause_overlap_seconds=overlap,
            source_bracketed_running_intervals=int(selected.sum()),
            flagged_fraction=float(flags[selected].mean()) if selected.any() else None,
            continuous_context_candidate=bool(end-offset==window_seconds and overlap==0 and selected.any()),
            validated_for_dfa=False))
    return dict(windows=windows,
        missing_preceding_record_intervals=int(np.isnan(lower).sum()),
        missing_following_record_intervals=int(np.isnan(upper).sum()),
        missing_following_record_rr_seconds=float(values[np.isnan(upper)].sum()),
        source_order_stopped_intervals=int((timer=='stopped').sum()),
        source_order_stopped_rr_seconds=float(values[timer=='stopped'].sum()),
        after_final_stop_source_order_intervals=sum(len(rr) for state,(at,rr) in zip(states,batches)
            if state=='stopped' and at is not None and stopped_at is not None and (at-start).total_seconds()>=stopped_at),
        interpretation='Message brackets and timer order only; no corrected/rescaled/stitched RR, '
                       'no exact beat timestamps, clinical artifact rate or DFA readiness claim')


def inspect(payload):
    counts, devices, laps, records, events, batches = Counter(), [], [], [], [], []
    session, last_record = {}, None
    following, states, pending, timer_state = [], [], [], 'unknown'
    session_fields = ('start_time', 'total_elapsed_time', 'total_timer_time', 'total_distance',
        'avg_heart_rate', 'max_heart_rate', 'total_training_effect',
        'total_anaerobic_training_effect', 'workout_rpe', 'workout_feel')
    lap_fields = ('start_time', 'total_timer_time', 'total_distance', 'avg_heart_rate',
                  'max_heart_rate', 'lap_trigger', 'intensity')
    with fitdecode.FitReader(io.BytesIO(payload)) as reader:
        for frame in reader:
            if frame.frame_type != fitdecode.FIT_FRAME_DATA:
                continue
            v = {field.name: field.value for field in frame.fields}
            counts[frame.name] += 1
            if frame.name == 'session':
                session = {k: v.get(k) for k in session_fields}
            elif frame.name == 'record':
                last_record = v.get('timestamp')
                if last_record is not None:
                    for index in pending:
                        following[index] = last_record
                    pending.clear()
                records.append((last_record, v.get('heart_rate')))
            elif frame.name == 'hrv':
                raw = v.get('time', ())
                raw = raw if isinstance(raw,(tuple,list)) else (raw,)
                batches.append((last_record, [x for x in raw if x is not None]))
                following.append(None)
                states.append(timer_state)
                pending.append(len(batches)-1)
            elif frame.name == 'lap':
                laps.append({k: v.get(k) for k in lap_fields})
            elif frame.name == 'device_info':
                d = {k: v.get(k) for k in ('garmin_product', 'antplus_device_type', 'source_type')}
                if d not in devices:
                    devices.append(d)
            elif frame.name == 'event' and v.get('event') == 'timer':
                events.append({k: v.get(k) for k in ('timestamp', 'event_type')})
                if v.get('event_type') == 'start':
                    timer_state = 'running'
                elif v.get('event_type') in {'stop','stop_all','stop_disable','stop_disable_all'}:
                    timer_state = 'stopped'
    if not session.get('start_time'):
        raise ValueError('Missing session start')
    start = session['start_time']
    rr = np.array([x for _, values in batches for x in values])
    if not len(rr):
        raise ValueError('No RR intervals')
    anchors = np.array([(at-start).total_seconds() if at else np.nan
                        for at, values in batches for _ in values])
    screened = screen_rr(rr)
    windows = []
    # Anchors are wall-clock offsets. Timer duration truncates paused activities.
    for offset in range(0, int(np.ceil(session['total_elapsed_time'])), 120):
        selected = (anchors >= offset) & (anchors < offset+120)
        if not selected.any():
            continue
        windows.append(dict(approx_start_s=offset, interval_count=int(selected.sum()),
            flagged_fraction=float(screened[selected].mean())))
    cumulative = 0.
    offsets = []
    for at, values in batches:
        cumulative += sum(values)
        if at:
            offsets.append(cumulative-(at-start).total_seconds())
    valid_records = [(at, hr) for at, hr in records if at and hr is not None]
    rt = np.array([(at-start).total_seconds() for at, _ in valid_records])
    hr = np.array([value for _, value in valid_records])
    for i, lap in enumerate(laps):
        lap['number'] = i+1
        meters, seconds = lap['total_distance'], lap['total_timer_time']
        lap['pace_min_mile'] = seconds / 60 / (meters / 1609.344) if meters else None
    return dict(fit_sha256=sha256(payload).hexdigest(), message_counts=dict(counts),
        session=session, devices=devices, timer_events=events, laps=laps,
        decoded_feedback=feedback_from_session_fields(session),
        rr_context=rr_context_windows(batches,following,states,events,start,session['total_elapsed_time']),
        rr=dict(interval_count=len(rr), sum_seconds=float(rr.sum()),
            min_seconds=float(rr.min()), max_seconds=float(rr.max()),
            below_250ms=int((rr < .25).sum()), above_2000ms=int((rr > 2.).sum()),
            screening=[dict(relative_threshold=t, flagged_count=int(screen_rr(rr,t).sum()),
                            flagged_fraction=float(screen_rr(rr,t).mean())) for t in (.10,.20,.30)],
            approximate_two_minute_windows=windows,
            cumulative_rr_minus_record_seconds_quantiles=np.percentile(offsets,[0,5,50,95,100]).tolist()),
        recorded_hr=dict(record_count=len(records), known_hr_records=len(valid_records),
            inter_record_seconds_quantiles=np.percentile(np.diff(rt),[0,50,95,100]).tolist(),
            integrated_beat_estimate=float(np.trapezoid(hr/60,rt))),
        interpretation='Screening flags are not validated artifact percentages or a diagnosis. '
            'Twenty-percent local-median deviation and 250–2000 ms bounds are diagnostic choices. '
            'FIT HRV lacks timestamps: preceding smart-record anchors are approximate; no rescaling '
            'or stitching has been performed. No DFA value or recovery threshold is published. '
            'Raw session RPE/feel fields require verified scale semantics before use. '
            'Integrated HR is smoothed/sparsely sampled, not a ground-truth beat count.')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    if args.source.suffix.lower() == '.zip':
        members = fit_members(args.source.read_bytes())
        if len(members) != 1:
            parser.error('Expected exactly one FIT member')
        _, payload = members[0]
    else:
        payload = args.source.read_bytes()
    report = inspect(payload)
    encoded = json.dumps(report, default=lambda x: x.isoformat(), indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(encoded+'\n')
    print(json.dumps(dict(session=report['session'], rr=report['rr'],
                         recorded_hr=report['recorded_hr']), default=str, indent=2))


if __name__ == '__main__':
    main()
