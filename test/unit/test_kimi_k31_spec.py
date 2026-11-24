# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3.1 spec routes one request's verify steps to Kimi-K3's launches and
wider ones to the wide launch."""

import sgl_amd_mono
from sgl_amd_mono.kimi_k3 import MAX_ROWS
from sgl_amd_mono.kimi_k31 import spec as k31
from sgl_amd_mono.kimi_k31 import wide


def test_registered_under_the_k31_architecture():
    assert (
        sgl_amd_mono.SPECS[("KimiK31ForConditionalGeneration", None)]
        == "sgl_amd_mono.kimi_k31.spec:SPEC"
    )


def test_the_narrow_and_wide_widths_meet_without_a_gap():
    # _Built routes rows up to NARROW to Kimi-K3's launches, which are built up to
    # MAX_ROWS; a width between the two would route to a launch never built
    assert wide.NARROW == MAX_ROWS
    widths = k31.SPEC.widths("gfx950")
    assert tuple(range(1, MAX_ROWS + 1)) == widths[:MAX_ROWS]
    assert all(w > wide.NARROW for w in widths[MAX_ROWS:])


def test_every_wide_width_has_a_bucket_at_each_draft_length():
    # the binding serves a width at every L it is a multiple of: an 8-token width
    # that reads as more 6-token requests than the largest bucket holds raised at
    # the launch
    top = max(wide.REQ_BUCKETS)
    for w in wide.wide_widths():
        for L in wide.SUPPORTED_L:
            if w % L == 0:
                assert w // L <= top, (w, L)


def test_the_launches_name_their_server_args():
    server_args = k31.SPEC.requires.server_args
    assert server_args["enable_linear_replayssm_spec"] == frozenset({True})
    assert server_args["speculative_algorithm"] == frozenset({"DSPARK", "DFLASH"})
