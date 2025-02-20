"""
Script to initialize the foundation models for the MACE project.
The default dowload dir is ~/.cache/mace/.
"""

from mace.calculators.foundations_models import mace_mp, mace_off

mp_list = ["small",
            "medium",
            "large",
            "medium-mpa-0",
            "small-0b",
            "medium-0b",
            "small-0b2",
            "medium-0b2",
            "medium-0b3",
            "large-0b2",
            "medium-omat-0",]
off_list  = ["small", "medium", "large"]

for mp in mp_list:
    mace_mp(mp)

for off in off_list:
    mace_off(off)
