# Reference data for the ray-wave reproduction probes

Binary inputs, not code and not generated output. They are committed because
they are small, are inputs to a reproduction, and cannot be regenerated from
anything in this repository.

| File | Shape / dtype | What it is |
| -- | -- | -- |
| `demo2_smile_phase_profile.npy` | `(100, 100)` float32, values in `[-pi, pi]` | SLM phase mask for the planar free-space hologram system (Cheng et al., ACS Photonics 2026, DOI 10.1021/acsphotonics.6c00818, Fig 5b) |
| `demo3_smile_phase_profile.npy` | `(200, 200)` float32, values in `[-pi, pi]` | SLM phase mask for the hologram-plus-refractive-lens system (same paper, Fig 5c) |

Provenance: taken from the paper's reference implementation as **data**. No
upstream source code is vendored here. The coupler source manifests under
`knowledge/couplers/*/source_manifest.yaml` carry the corresponding
`source_type: reference_data` records.

Note on `.gitignore`: the repository ignores `data/` globally as a
generated-dataset pattern. This directory is re-included by an explicit
negation, because these files are inputs rather than outputs.
