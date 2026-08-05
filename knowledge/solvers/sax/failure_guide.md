# SAX failure guide

Real issues hit while building this knowledge pack (2026-07-30), with
repairs. Add to this file rather than silently working around a new one.

## Port names don't match the online docs example

**Symptom:** following the `gdsfactory.github.io/sax` quickstart's port
names (`in0`, `out0`) against the installed 0.18.2 package raises a
`KeyError` or produces an empty/wrong S-matrix entry.

**Cause:** the installed package's own docstring examples call
`sax.set_port_naming_strategy("optical")` first, which changes built-in
model port names to `o1`/`o2`/`o3`/`o4`. The online docs page was not
regenerated for this. See `conventions.md`.

**Fix:** call `sax.set_port_naming_strategy("optical")` (or check
`sax.get_port_naming_strategy()`) before assuming any port name scheme, and
read port names off the actual returned S-matrix dict's keys rather than
hardcoding them.

## `TypeError: Object of type ArrayImpl is not JSON serializable`

**Symptom:** raised when trying to `json.dumps` a dict containing values
pulled directly out of a SAX `SDict` S-matrix.

**Cause:** S-matrix dict values are JAX array scalars (`jax.Array`), not
bare Python `complex`.

**Fix:** wrap with `complex(x)` before using outside JAX (JSON
serialization, `cmath` functions, equality against a Python literal for a
sanity check).

## `pip install sax` alone does not need a namesquat workaround (unlike chromatix)

**Not a failure** -- documented here to prevent an agent from over-applying
the chromatix lesson. `sax` on PyPI IS the real gdsfactory-team package;
the only real hazard is stale documentation citing the old
`flaport/sax` repository URL (it 301-redirects, so it still resolves, but
should not be cited as canonical going forward) and using git tags as a
version source (they lag ~9 major/minor versions behind PyPI).

## Python version

SAX 0.18.2 requires Python `>=3.11.0`. This project's own `pyproject.toml`
requires `>=3.11` (CLAUDE.md section 11 says the same), so there is no
project-level conflict, but the host machine used to develop this knowledge
pack only had Python 3.10/3.8 available -- the `agent_solver` Docker image
(python:3.12-slim) is what actually satisfies this. Do not assume a host
`.venv` is new enough without checking.

## `docker run` heredoc probes need `-i`

**Symptom:** `docker run --rm agent_solver python3 << 'EOF' ... EOF` prints
nothing and exits silently.

**Cause:** `docker run` without `-i` does not attach the container's stdin,
so the heredoc content is never delivered to the `python3` process inside
the container.

**Fix:** add `-i`: `docker run --rm -i agent_solver python3 << 'EOF' ...`.
(Not SAX-specific, but was hit repeatedly while developing these probes and
is worth recording once.)
