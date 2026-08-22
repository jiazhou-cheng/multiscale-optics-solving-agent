"""Scientific studies built on the model/coupler stack.

A study is a specific investigation with its own controller, candidate model and
oracle -- not a reusable capability. Keeping them out of `couplers/` and
`solvers/` is what stops a one-system result from being read as a general one.
"""
