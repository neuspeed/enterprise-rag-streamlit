"""Read-only query API over the built-in store.

Serves retrieval only: it ranks and cites fragments, it never invokes a model
to answer. Answering is the agent role, which is configured separately and
reuses this package's stores.
"""
