"""Shared score/diversity retention, extracted from the developed planner beam."""


def retain_diverse(values,limit,*,rank,identity,signature):
    if limit<1:
        raise ValueError('Positive beam width required')
    ranked=sorted(values,key=rank)
    if len(ranked)<=limit:
        return ranked
    count=max(1,limit//2)
    selected=ranked[:count]
    identities={identity(v) for v in selected}
    signatures={signature(v) for v in selected}
    for value in ranked[count:]:
        key=signature(value)
        if key in signatures:
            continue
        selected.append(value)
        identities.add(identity(value))
        signatures.add(key)
        if len(selected)==limit:
            break
    if len(selected)<limit:
        selected.extend(v for v in ranked if identity(v) not in identities)
    return sorted(selected[:limit],key=rank)
