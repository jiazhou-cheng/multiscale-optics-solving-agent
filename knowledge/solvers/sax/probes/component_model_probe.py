"""Single-component S-matrix probe for M_CIRCUIT_SAX.

Exercises two built-in models (`coupler_ideal`, `straight`) directly
(no circuit/netlist assembly) and checks their S-matrices against physical
invariants: energy conservation, reciprocity, and -- for `straight` -- an
analytic propagation-phase oracle. Every port name below depends on
`sax.set_port_naming_strategy("optical")`, which is NOT the default; see
conventions.md.

Run inside the agent_solver container:
    ./run.sh python knowledge/solvers/sax/probes/component_model_probe.py
"""

from __future__ import annotations

import cmath
import json
import math

import sax
import sax.models as sm


def main() -> None:
    sax.set_port_naming_strategy("optical")

    wavelength = 1.55
    coupling = 0.5
    s = sm.coupler_ideal(wl=wavelength, coupling=coupling)
    thru = complex(s[("o1", "o4")])
    cross = complex(s[("o1", "o3")])

    neff, ng, length, wl0 = 2.34, 3.4, 10.0, 1.55
    s2 = sm.straight(wl=wavelength, wl0=wl0, neff=neff, ng=ng, length=length, loss_dB_cm=0.0)
    t = complex(s2[("o1", "o2")])
    phase_observed = cmath.phase(t)
    phase_analytic = (2 * math.pi * neff * length / wavelength) % (2 * math.pi)

    report = {
        "coupler_ideal": {
            "wavelength": wavelength,
            "coupling": coupling,
            "thru_o1_o4": [thru.real, thru.imag],
            "cross_o1_o3": [cross.real, cross.imag],
            "energy_conservation": abs(thru) ** 2 + abs(cross) ** 2,
            "reciprocal_o4_o1_equals_o1_o4": bool(complex(s[("o4", "o1")]) == thru),
        },
        "straight": {
            "wavelength": wavelength,
            "neff": neff,
            "length": length,
            "magnitude_observed": abs(t),
            "phase_observed": phase_observed,
            "phase_analytic_exp_plus_i_k_L": phase_analytic,
            "phase_relative_error": abs(phase_observed - phase_analytic) / phase_analytic,
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
