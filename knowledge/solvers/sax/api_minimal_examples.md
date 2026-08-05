# Minimal SAX examples (validated against installed version `0.18.2`)

Every snippet below was actually executed inside the `agent_solver`
container against the pinned install (`sax==0.18.2` in
`docker/requirements.txt`). Output values shown are real, captured on
2026-07-30, not illustrative.

## 1. Import / initialization

```python
import sax
sax.__version__   # -> "0.18.2"
```

Full probe: `probes/import_probe.py`; captured output:
`expected/import_probe.json`.

## 2. Single component model (no circuit assembly)

```python
import sax, sax.models as sm

sax.set_port_naming_strategy("optical")   # NOT the default -- see conventions.md

s = sm.coupler_ideal(wl=1.55, coupling=0.5)
thru = complex(s[("o1", "o4")])   # 0.7071067811865476
cross = complex(s[("o1", "o3")])  # 0.7071067811865476j

s2 = sm.straight(wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss_dB_cm=0.0)
t = complex(s2[("o1", "o2")])     # magnitude 1.0, phase 0.6080501910173587
```

Values returned in the S-matrix dict are JAX array scalars, not bare Python
complex -- wrap with `complex(...)` before using them outside JAX (e.g. for
JSON serialization or `cmath` functions).

Full probe: `probes/component_model_probe.py`; captured output:
`expected/component_model_probe.json`.

## 3. Full netlist circuit (real Mach-Zehnder interferometer)

```python
import sax, sax.models as sm

sax.set_port_naming_strategy("optical")

netlist = {
    "instances": {
        "c1": "coupler_ideal", "wg_short": "straight",
        "wg_long": "straight", "c2": "coupler_ideal",
    },
    "connections": {
        "c1,o3": "wg_short,o1", "c1,o4": "wg_long,o1",
        "wg_short,o2": "c2,o1", "wg_long,o2": "c2,o2",
    },
    "ports": {"in0": "c1,o1", "in1": "c1,o2", "out0": "c2,o3", "out1": "c2,o4"},
}
models = {"coupler_ideal": sm.coupler_ideal, "straight": sm.straight}
mzi, info = sax.circuit(netlist, models)

result = mzi(
    wl=1.55,
    c1={"coupling": 0.5}, c2={"coupling": 0.5},
    wg_short={"length": 10.0, "neff": 2.34, "wl0": 1.55, "ng": 3.4, "loss_dB_cm": 0.0},
    wg_long={"length": 15.0, "neff": 2.34, "wl0": 1.55, "ng": 3.4, "loss_dB_cm": 0.0},
)
power_00 = abs(complex(result[("in0", "out0")])) ** 2   # -> 0.9770696282000273
```

Matched against the analytic MZI transmission formula
`sin^2(pi * n_eff * dl / wavelength)` = `0.977069628200026` (relative error
1.36e-15); energy conservation and reciprocity both hold exactly across the
full 2x2 circuit. Full probe: `probes/circuit_probe.py`; captured output:
`expected/circuit_probe.json`.

## 4. Batched / vectorized example

Not yet executed in this repository. `sax.wl_c()`/similar helpers and JAX
broadcasting suggest `wl` can be an array for a wavelength sweep (used in
the package's own docstring examples with `matplotlib` plots), but this has
not been independently verified here.

## 5. Gradient example

```python
import jax, jax.numpy as jnp
import sax, sax.models as sm
sax.set_port_naming_strategy("optical")

def thru_power(coupling):
    s = sm.coupler_ideal(wl=1.55, coupling=coupling)
    return jnp.abs(s[("o1", "o4")]) ** 2

jax.grad(thru_power)(0.3)   # -> -1.0000000000000002
```

Centered finite difference (`eps=1e-4`) gives `-1.000000000000445`
(relative error 4.4e-13). This is a closed-form trig expression -- tight
agreement is expected and does NOT validate differentiability through the
`sax.circuit` matrix-solve path, which has not been gradient-probed. Full
probe: `probes/gradient_probe.py`; captured output:
`expected/gradient_probe.json`.

## 6. Serialization / export

`sax.write_touchstone`/`sax.parse_touchstone` and `sax.parse_lumerical_dat`
exist for interop with measurement/instrument data. Not yet exercised.

## 7. Common error signatures and repairs

See `failure_guide.md`.
