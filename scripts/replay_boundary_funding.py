"""Diagnostic daily/upload trial; inherited proposals and gates remain unchanged.

Use the ordinary replay arguments. This separate experiment entry point is not
a new application preference or default enablement of the scoped prototype.
Fail fast on the original gates, retaining independent delivery audit/provenance.
"""
from scripts.replay_synthetic_easy_dose import main


if __name__=='__main__':
    main(funding_prototype=True)
