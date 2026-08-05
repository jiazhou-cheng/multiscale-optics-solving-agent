"""Netlist circuit-composition probe for M_CIRCUIT_SAX.

Builds a real Mach-Zehnder interferometer (MZI) from two `coupler_ideal`
instances and two `straight` waveguides via `sax.circuit`'s netlist DSL,
and checks the assembled S-matrix against:
  1. an analytic MZI transmission formula (independent oracle), and
  2. energy conservation / reciprocity across the full 2x2 circuit.

This exercises the full circuit-assembly path, not just an isolated
component model (see component_model_probe.py for that).

Run inside the agent_solver container:
    ./run.sh python knowledge/solvers/sax/probes/circuit_probe.py
"""

from __future__ import annotations

import json
import math

import sax
import sax.models as sm

NETLIST = {
    "instances": {
        "c1": "coupler_ideal",
        "wg_short": "straight",
        "wg_long": "straight",
        "c2": "coupler_ideal",
    },
    "connections": {
        "c1,o3": "wg_short,o1",
        "c1,o4": "wg_long,o1",
        "wg_short,o2": "c2,o1",
        "wg_long,o2": "c2,o2",
    },
    "ports": {
        "in0": "c1,o1",
        "in1": "c1,o2",
        "out0": "c2,o3",
        "out1": "c2,o4",
    },
}
MODELS = {"coupler_ideal": sm.coupler_ideal, "straight": sm.straight}


def main() -> None:
    sax.set_port_naming_strategy("optical")

    mzi, info = sax.circuit(NETLIST, MODELS)

    wavelength = 1.55
    neff, ng = 2.34, 3.4
    length_short = 10.0
    length_long = 15.0
    dl = length_long - length_short

    result = mzi(
        wl=wavelength,
        c1={"coupling": 0.5},
        c2={"coupling": 0.5},
        wg_short={"length": length_short, "neff": neff, "wl0": wavelength, "ng": ng, "loss_dB_cm": 0.0},
        wg_long={"length": length_long, "neff": neff, "wl0": wavelength, "ng": ng, "loss_dB_cm": 0.0},
    )

    t00 = complex(result[("in0", "out0")])
    t01 = complex(result[("in0", "out1")])

    dphi = 2 * math.pi * neff * dl / wavelength
    t_analytic_sin2 = math.sin(dphi / 2) ** 2

    report = {
        "netlist_ports": list(NETLIST["ports"].keys()),
        "result_port_pairs_sample": sorted(str(k) for k in result.keys())[:6],
        "in0_to_out0": {"real": t00.real, "imag": t00.imag, "power": abs(t00) ** 2},
        "in0_to_out1": {"real": t01.real, "imag": t01.imag, "power": abs(t01) ** 2},
        "energy_conservation_from_in0": abs(t00) ** 2 + abs(t01) ** 2,
        "reciprocity_out0_in0_equals_in0_out0": bool(complex(result[("out0", "in0")]) == t00),
        "analytic_T_out0_sin2_form": t_analytic_sin2,
        "relative_error_vs_analytic": abs(abs(t00) ** 2 - t_analytic_sin2) / t_analytic_sin2,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
