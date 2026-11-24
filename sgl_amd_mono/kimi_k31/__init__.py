# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 monokernels (RFC #43949).

Kimi-K3.1 is Kimi-K3 with DSA full-attention layers, and every shape the Kimi-K3
launches build in is the same. Steps of up to 8 rows run those launches
(``kimi_k3``); a speculative verify step of R >= 2 requests x L draft tokens
above them runs one persistent launch a KDA layer (``wide``,
``gfx950/layer.py``). ``spec`` routes both; SGLang runs every other step.
"""
