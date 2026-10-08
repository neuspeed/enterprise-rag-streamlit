"""The answer agent: retrieval plus one grounded, cited completion.

Everything here is shared between the two surfaces — the SSE chat endpoint and
the Streamlit chat tab — so the "answer only from these fragments" contract is
written once and neither surface can drift from it.
"""
