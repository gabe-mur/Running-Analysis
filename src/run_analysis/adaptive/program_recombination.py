"""Physical prefix/suffix candidates, with no transferred score or date lock."""
from itertools import permutations


def recombinations(programs, cuts):
    programs=tuple(dict.fromkeys(tuple(sorted(p)) for p in programs))
    if any(len(dict(p))!=len(p) for p in programs):
        raise ValueError('Recombination parents cannot contain duplicate dates')
    seen=set(programs)
    # Visit every cut for a parent pair before advancing to the next pair, rather
    # than spending the cap entirely at one boundary. Both directions are priced.
    for left,right in permutations(programs,2):
        for cut in sorted(set(cuts)):
            p=tuple(sorted((*((d,k) for d,k in left if d<cut),
                            *((d,k) for d,k in right if d>=cut))))
            if p not in seen:
                seen.add(p)
                yield p,dict(cut=cut,prefix=left,suffix=right)
