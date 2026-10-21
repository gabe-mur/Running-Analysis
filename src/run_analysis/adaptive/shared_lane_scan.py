"""Fair, bounded neighborhood scans; selection remains with each caller.

One proposal per active lane per round. An incomplete scan is never a local
certificate; cached rejections are shared exactly like cached eligible results.
"""


def scan_lanes(streams, evaluate, *, cache, budget):
    if budget < 1:
        raise ValueError('Positive neighborhood budget required')
    states = [dict(iterator=iter(stream), complete=False, blocked=False,
                   examined=0, evaluations=0, cache_hits=0, results=[])
              for stream in streams]
    while any(not s['complete'] and not s['blocked'] for s in states):
        for state in states:
            if state['complete'] or state['blocked']:
                continue
            try:
                key, proposal = next(state['iterator'])
            except StopIteration:
                state['complete'] = True
                continue
            if key not in cache:
                if len(cache) >= budget:
                    state['blocked'] = True
                    continue
                cache[key] = evaluate(proposal)
                state['evaluations'] += 1
            else:
                state['cache_hits'] += 1
            state['examined'] += 1
            if cache[key] is not None:
                state['results'].append(cache[key])
    return [{k: v for k, v in state.items() if k != 'iterator'} for state in states]
