"""Partition already supported work windows; never classify intensity from HR."""


def phase_layout(windows,elapsed):
    if not windows or elapsed<=0:
        return ()
    result,at=[],0.
    for start,end in windows:
        if not 0<=at<=start<end<=elapsed:
            raise ValueError('Ordered in-session work windows required')
        if start>at:
            result.append(('nonwork',at,start))
        result.append(('work',start,end))
        at=end
    if at<elapsed:
        result.append(('nonwork',at,elapsed))
    return tuple(result)


def detailed_layout(layout):
    return tuple((kind if kind=='work' else
        'warmup' if i==0 else 'cooldown' if i==len(layout)-1 else 'recovery',a,b)
        for i,(kind,a,b) in enumerate(layout))
