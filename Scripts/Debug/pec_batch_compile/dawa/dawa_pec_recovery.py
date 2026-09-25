"""Generate a synthetic DAWA subject and fit it with the compiled PEC GPU sampler.

The shared pipeline in dawa_pec_fit.py also fits recorded behavioral data.
Recovery keeps generation, optimization, and validation seeds separate.
"""

from dawa_pec_fit import main


if __name__ == "__main__":
    main(recovery=True)
