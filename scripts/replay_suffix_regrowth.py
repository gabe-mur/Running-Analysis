"""Offline bounded suffix-regrowth trial in the real daily/upload workflow."""
from scripts.replay_synthetic_easy_dose import main

if __name__=='__main__':
    main(funding_prototype=True,regrowth_prototype=True)
