# SPDX-License-Identifier: Apache-2.0
"""The asm decode layers against SGLang's decode path through the whole model (on / off):
the same greedy batches on servers that differ only in the path of the decode steps,
compared on the output tokens, their logprobs and the last layer's residual (the 4 x 5120
values that SGLang returns as the hidden state: pre_hc_head, before the head's mHC
combine). So the drift over the 40 layers and the asm K1 are in the result;
check_asm_layer.py --mode check compares each wide layer's K2 with SGLang's FFN half on
the same inputs, which has neither.

Roles, one eager server each, with the control's flags (check_asm_layer.py):
  std      SGLang's decode path at every width (no --enable-amd-monokernel)
  ctrl     the control: the monokernel up to 48 rows, SGLang's path above
  ctrl2    ctrl again on a new server: the run-to-run noise of the same path
  wide     ctrl with the asm decode layers above 48 rows (SGLANG_AMD_MONO_ASM_WIDE=1, K2 v3,
           bf16 wire, SGLang's attention half)
  wide_v1  wide with K2 v1 (SGLANG_AMD_MONO_ASM_K2=v1, the plugin's default)
  k1       wide_v1 with the asm K1 (SGLANG_AMD_MONO_ASM_K1=v3)
Each row count is one batch: a 1-token pass fills the prefix cache, then the measured pass
(--steps greedy tokens: a short prefill step, then decode steps of all requests together)
returns the token logprobs and the hidden state of the first --hidden requests at their
last step. Every decode step's #running-req is in the server log (--decode-log-interval 1).
The servers run --return-hidden-states-mode full and the requests ask with True, because
the image's HTTP parser (sglang-api-types BoolOrList) takes only booleans, and True asks
for every forward's rows; after the cache fill that is one prefill row and one row a decode
step, and the test keeps the last row (batch_result_processor.py, output_streamer.py).
A role fails unless SGLang's startup line routes exactly its planned row counts to the
monokernel forward (the wide roles: all, ctrl and ctrl2: up to 48, std: none).
Prompts (--prompts): gsm8k (default), natural text in SGLang's few_shot_gsm8k format
(--shots solved questions, then a question), tokenized here and sent as token ids, so that
a batch reaches the scheduler as one message; random, check_asm_layer.py's random tokens,
on which greedy tokens flip at the smallest change.

--compare (the FP8-shared-expert reference, gated up to 64
rows): at 32 and 48 rows every role but std runs the monokernel, so they must agree within
the run-to-run noise (ctrl2 against ctrl; without ctrl2 exactly: the same tokens, the
residual within ID_REL). The asm pairs (wide, wide_v1 and k1 against ctrl) at 49-64 rows
must keep the residual within RES_REL (vLLM's residual bound) while the tokens agree, and
at most a quarter of the requests may change a token; this gate applies only without the
shared-expert fusion, because with it SGLang's shared expert runs requantized to MXFP4 and
K2 the checkpoint's FP8. Above 64 rows SGLang's router logits come from a bf16 GEMM instead
of the split-K GEMV's fp32 sum (rocm_linear_utils.py, router_gemv_hip.py
ROCM_ROUTER_MAX_TOKENS), so near-tie routing differs from K2's by construction: reported.
Reported without a bound: the noise, the floor (ctrl against std at every row count: the
monokernel at 32 and 48 rows; above, the mHC boundary of the decode steps SGLang runs,
fused with the monokernel on and Mega-mHC's seam with it off,
SGLANG_OPT_DSV41_MEGA_MHC_PREFILL=1, deepseek_v4_hip.py fused_boundary_this_step), wide
against wide_v1 (v3's part) and k1 against wide_v1 (the asm K1's part).

    python check_asm_onoff.py --role ctrl --out DIR [--rows 32,48,64,128,192,256]
                              [--no-shared-fusion] [--prompts gsm8k|random]
    python check_asm_onoff.py --compare --out DIR
--no-shared-fusion drops the control's --enforce-shared-experts-fusion, so SGLang's shared
expert runs the checkpoint's FP8, as the monokernel and K2 do, not MXFP4; every role of an
--out directory must be run alike.
Writes DIR/<role>.json, DIR/<role>_hidden.npz, DIR/<role>.server.{out,err} and
DIR/compare.json.
"""

import argparse
import hashlib
import json
import re
import signal
import sys
import time
from collections import Counter
from pathlib import Path

from check_asm_layer import CONTROL_FLAGS, MODEL, model_path, prompts

ROLES = ("std", "ctrl", "ctrl2", "wide", "wide_v1", "k1")
WIDE_ROLES = ("wide", "wide_v1", "k1")
PAIRS = (("ctrl2", "ctrl"), ("ctrl", "std"), ("wide", "ctrl"), ("wide_v1", "ctrl"), ("k1", "ctrl"),
         ("wide", "wide_v1"), ("k1", "wide_v1"))  # fmt: skip
# the plugin's asm switches at their defaults, as the control runs (spec.py, asm/layer.py, asm/k2.py)
ASM_OFF = dict(SGLANG_AMD_MONO_ASM_WIDE="0", SGLANG_AMD_MONO_ASM_WIRE="bf16", SGLANG_AMD_MONO_ASM_K2="v1",
               SGLANG_AMD_MONO_ASM_K1="0", SGLANG_AMD_MONO_ASM_CHECK="")  # fmt: skip
MONO_MAX, WIDE_MAX = 48, 256  # the widest monokernel and wide-path steps (spec.py)
ROUTER_MAX = 64  # SGLang's split-K router GEMV: up to 64 rows (router_gemv_hip.py)
HC, HIDDEN = 4, 5120  # residual streams and their width (config hc_mult, hidden_size)
RES_REL, ID_REL = 0.02, 1e-3
GSM8K_DATA = "/int_logs/data/gsm8k_test.jsonl"


def routed(line) -> set:
    """The batch sizes of SGLang's startup line that routes steps to the monokernel forward."""
    if not line or "batch size" not in line:
        return set()
    return {
        int(x)
        for x in re.findall(r"\d+", line.split("batch size", 1)[-1].split("]")[0])
    }


def kind(a: str, b: str, m: int) -> str:
    if (a, b) == ("ctrl2", "ctrl"):
        return "noise"
    if (a, b) == ("ctrl", "std"):
        return "floor"
    if m <= MONO_MAX:
        return "same"
    if b == "ctrl":
        return "asm" if m <= ROUTER_MAX else "asm above 64"
    return "v3 part" if a == "wide" else "k1 part"


def gsm8k_texts(path: str, n: int, shots: int) -> list:
    """n prompts in SGLang's few_shot_gsm8k format (get_one_example, get_few_shot_examples):
    prompt i holds the solved questions i+1 .. i+shots, then question i."""
    lines = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

    def one(i: int, answer: bool) -> str:
        q = lines[i % len(lines)]
        return (
            "Question: "
            + q["question"]
            + "\nAnswer:"
            + (" " + q["answer"] if answer else "")
        )

    return ["".join(one(i + 1 + j, True) + "\n\n" for j in range(shots)) + one(i, False)
            for i in range(n)]  # fmt: skip


def tokenize(model: str, texts: list) -> list:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path(model), trust_remote_code=True)
    return [tok.encode(t) for t in texts]


def serve(args, out: Path) -> dict:
    import numpy as np
    import requests
    from sglang.test.test_utils import (
        popen_launch_server,
        terminate_and_kill_process_tree,
    )

    url = f"http://127.0.0.1:{args.port}"
    rows = [int(r) for r in args.rows.split(",")]
    # every role draws the same WIDE_MAX prompts (prompts(n) draws n lengths first), before the server starts
    n = max(WIDE_MAX, *rows)
    if args.prompts == "gsm8k":
        p = tokenize(args.model, gsm8k_texts(args.data, n, args.shots))
        source = dict(prompts="gsm8k", shots=args.shots,
                      data_sha1=hashlib.sha1(Path(args.data).read_bytes()).hexdigest()[:8])  # fmt: skip
    else:
        p, source = prompts(n), dict(prompts="random")
    # popen_launch_server adds env to os.environ, so every switch is set, not left out
    env = dict(ASM_OFF)
    if args.role in WIDE_ROLES:
        env.update(SGLANG_AMD_MONO_ASM_WIDE="1", SGLANG_AMD_MONO_ASM_K2="v3" if args.role == "wide" else "v1",
                   SGLANG_AMD_MONO_ASM_K1="v3" if args.role == "k1" else "0")  # fmt: skip
    drop = {"--enable-amd-monokernel"} if args.role == "std" else set()
    if args.no_shared_fusion:
        drop.add("--enforce-shared-experts-fusion")
    flags = [f for f in CONTROL_FLAGS if f not in drop]
    flags += ["--watchdog-timeout", "600", "--cuda-graph-backend-decode", "disabled",
              "--return-hidden-states-mode", "full", "--decode-log-interval", "1"]  # fmt: skip
    base = out / f"{args.role}.server"
    logs = (open(f"{base}.out", "w+"), open(f"{base}.err", "w+"))
    t0 = time.time()
    proc = popen_launch_server(
        model_path(args.model), url, timeout=3600,
        other_args=flags, env=env, return_stdout_stderr=logs,
    )  # fmt: skip
    res = dict(role=args.role, rows=rows, steps=args.steps, flags=flags, **source,
               shared_fusion=not args.no_shared_fusion, asm_env=env,
               startup_s=round(time.time() - t0, 1), batches={})  # fmt: skip
    hidden = {}

    def generate(body: dict) -> list:
        r = requests.post(url + "/generate", json=body, timeout=3600)
        if r.status_code != 200:
            raise RuntimeError(f"/generate answered {r.status_code}: {r.text[:1000]}")
        return r.json()

    try:
        greedy = {"temperature": 0.0, "ignore_eos": True}
        for m in rows:
            requests.post(url + "/flush_cache", params={"timeout": 30}, timeout=40)
            generate(
                {"input_ids": p[:m], "sampling_params": dict(greedy, max_new_tokens=1)}
            )
            k = min(m, args.hidden)
            t = time.time()
            outs = generate({
                "input_ids": p[:m],
                "sampling_params": dict(greedy, max_new_tokens=args.steps),
                "return_logprob": True,
                "return_hidden_states": [True] * k + [False] * (m - k),
            })  # fmt: skip
            meta = [o["meta_info"] for o in outs]
            res.setdefault("forwards", {})[str(m)] = len(meta[0]["hidden_states"])
            res["batches"][str(m)] = dict(
                wall_s=round(time.time() - t, 1),
                tokens=[list(o["output_ids"]) for o in outs],
                logprobs=[[lp[0] for lp in mi["output_token_logprobs"]] for mi in meta],
                cached=[mi.get("cached_tokens") for mi in meta],
                prompt=[len(x) for x in p[:m]],
            )
            # the last forward's last row: the last decode step
            hidden[f"h{m}"] = np.stack(
                [
                    np.asarray(mi["hidden_states"][-1], dtype=np.float32).reshape(-1)[
                        -HC * HIDDEN :
                    ]
                    for mi in meta[:k]
                ]
            )
    finally:
        terminate_and_kill_process_tree(proc)
        for f in logs:
            f.close()
    np.savez(out / f"{args.role}_hidden.npz", **hidden)
    log = Path(f"{base}.out").read_text(errors="replace") + Path(
        f"{base}.err"
    ).read_text(errors="replace")
    res["decode_rows"] = [
        int(x) for x in re.findall(r"Decode batch[^,\n]*, #running-req: (\d+)", log)
    ]
    res["routed_line"] = next(
        (
            ln.strip()
            for ln in log.splitlines()
            if "monokernel" in ln and " runs " in ln
        ),
        None,
    )
    res["memory_fault_lines"] = [
        ln.strip() for ln in log.splitlines() if "Memory access fault" in ln
    ][:4]
    return res


def compare(out: Path) -> dict:
    import numpy as np

    runs = {}
    for role in ROLES:
        if (out / f"{role}.json").exists():
            runs[role] = (
                json.loads((out / f"{role}.json").read_text()),
                np.load(out / f"{role}_hidden.npz"),
            )
    fused = any(r.get("shared_fusion", True) for r, _ in runs.values())
    res = dict(roles=list(runs), shared_fusion=fused, asm_gated=not fused,
               asm_rows=[MONO_MAX + 1, ROUTER_MAX], pairs=[], failures=[])  # fmt: skip
    fails = res["failures"]
    if len({r.get("shared_fusion", True) for r, _ in runs.values()}) > 1:
        fails.append("the roles differ in --enforce-shared-experts-fusion")
    if (
        len(
            {(r.get("prompts", "random"), r.get("data_sha1")) for r, _ in runs.values()}
        )
        > 1
    ):
        fails.append("the roles differ in --prompts or its data")
    for role, (r, _) in runs.items():
        seen = Counter(r["decode_rows"])
        fails += [f"{role}: {seen[m]} decode steps of {m} rows, not {r['steps'] - 1}"
                  for m in r["rows"] if seen[m] < r["steps"] - 1]  # fmt: skip
    noise = {}  # rows -> (requests changed, residual max) of ctrl2 against ctrl
    for a, b in PAIRS:
        if a not in runs or b not in runs:
            continue
        (ra, ha), (rb, hb) = runs[a], runs[b]
        if ra["steps"] != rb["steps"] or ra["decode_rows"] != rb["decode_rows"]:
            fails.append(
                f"{a} / {b}: not the same steps (--steps or the logged #running-req differ)"
            )
            continue
        steps = ra["steps"]
        for m in [x for x in ra["rows"] if x in rb["rows"]]:
            ba, bb = ra["batches"][str(m)], rb["batches"][str(m)]
            if ba["prompt"] != bb["prompt"]:
                fails.append(f"{a} / {b} at {m} rows: not the same prompts")
                continue
            # first step whose token differs (steps if none)
            div = [next((s for s, (x, y) in enumerate(zip(ta, tb)) if x != y), steps)
                   for ta, tb in zip(ba["tokens"], bb["tokens"])]  # fmt: skip
            lp = max((abs(x - y) for la, lb, d in zip(ba["logprobs"], bb["logprobs"], div)
                      for x, y in list(zip(la, lb))[:d]), default=0.0)  # fmt: skip
            rel, stream = [], []
            for i, (xa, xb) in enumerate(zip(ha[f"h{m}"], hb[f"h{m}"])):
                if div[i] < steps:
                    continue
                rel.append(float(np.linalg.norm(xa - xb) / np.linalg.norm(xb)))
                da, db = (xa - xb).reshape(HC, -1), xb.reshape(HC, -1)
                stream.append(
                    float(
                        (np.linalg.norm(da, axis=1) / np.linalg.norm(db, axis=1)).max()
                    )
                )
            k = kind(a, b, m)
            rec = dict(pair=f"{a}/{b}", rows=m, kind=k, gated=k == "same" or (k == "asm" and not fused),
                       requests=len(div), agree=sum(d == steps for d in div), first_token=sum(d == 0 for d in div),
                       first_diff=sorted(d for d in div if d < steps)[:8], logprob_max=lp, res_n=len(rel),
                       res_max=max(rel, default=None), res_median=float(np.median(rel)) if rel else None,
                       stream_max=max(stream, default=None))  # fmt: skip
            res["pairs"].append(rec)
            changed = rec["requests"] - rec["agree"]
            if k == "noise":
                noise[m] = (changed, rec["res_max"] or 0.0)
            if k == "same":
                n_changed, n_res = noise.get(m, (0, 0.0))
                if changed > 2 * n_changed or (rec["res_max"] or 0.0) > max(
                    ID_REL, 2 * n_res
                ):
                    fails.append(f"{a} / {b} at {m} rows (the same path): {changed} requests changed a token, "
                                 f"residual max {rec['res_max']}; noise {n_changed} and {n_res}")  # fmt: skip
            if k == "asm" and rec["gated"]:
                if not rel:
                    fails.append(
                        f"{a} / {b} at {m} rows: no request with a hidden state kept its tokens"
                    )
                elif rec["res_max"] >= RES_REL:
                    fails.append(
                        f"{a} / {b} at {m} rows: residual max {rec['res_max']:.4f}"
                    )
                if changed > rec["requests"] // 4:
                    fails.append(
                        f"{a} / {b} at {m} rows: {changed} of {rec['requests']} requests changed a token"
                    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=ROLES)
    ap.add_argument(
        "--compare", action="store_true", help="compare the roles found in --out"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="32,48,64,128,192,256")
    ap.add_argument(
        "--steps", type=int, default=8, help="tokens a request in the measured pass"
    )
    ap.add_argument(
        "--hidden",
        type=int,
        default=16,
        help="requests a batch that return the hidden state",
    )
    ap.add_argument(
        "--no-shared-fusion",
        action="store_true",
        help="without --enforce-shared-experts-fusion (SGLang's shared expert in FP8, as "
        "the monokernel and K2); every role of an --out directory alike",
    )
    ap.add_argument("--prompts", choices=("gsm8k", "random"), default="gsm8k")
    ap.add_argument("--data", default=GSM8K_DATA, help="GSM8K test set (jsonl)")
    ap.add_argument(
        "--shots",
        type=int,
        default=16,
        help="solved questions before a gsm8k prompt's question",
    )
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--port", type=int, default=30001, help="the slot's server port")
    args = ap.parse_args()
    if bool(args.role) == args.compare:
        ap.error("give --role or --compare")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.compare:
        res = compare(out)
        (out / "compare.json").write_text(json.dumps(res, indent=1))
        for p in res["pairs"]:
            res_txt = (
                "no request"
                if p["res_max"] is None
                else (
                    f"max {p['res_max']:.4f}, median {p['res_median']:.4f}, stream max {p['stream_max']:.4f}"
                )
            )
            print(f"{p['pair']:12s} {p['rows']:4d} rows ({p['kind']}{', gated' if p['gated'] else ''}): "
                  f"tokens agree {p['agree']}/{p['requests']} (first changes {p['first_diff']}, at the first token "
                  f"{p['first_token']}), logprob max {p['logprob_max']:.4f}, residual of {p['res_n']}: {res_txt}")  # fmt: skip
        print(f"asm pairs gated at {MONO_MAX + 1}-{ROUTER_MAX} rows: {res['asm_gated']} "
              f"(shared-expert fusion: {res['shared_fusion']})")  # fmt: skip
        fails = res["failures"]
    else:
        # a SIGTERM (timeout, docker stop) must run serve's finally, which stops the server
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        res = serve(args, out)
        (out / f"{args.role}.json").write_text(json.dumps(res))
        fails = [f"{m}-row batch: {sum(len(t) != args.steps for t in b['tokens'])} requests short of {args.steps} tokens"
                 for m, b in res["batches"].items() if any(len(t) != args.steps for t in b["tokens"])]  # fmt: skip
        if res["memory_fault_lines"]:
            fails.append("memory access fault in the server log")
        want = {
            m
            for m in res["rows"]
            if args.role in WIDE_ROLES
            or (args.role in ("ctrl", "ctrl2") and m <= MONO_MAX)
        }
        got = routed(res["routed_line"]) & set(res["rows"])
        if got != want:
            fails.append(
                f"row counts routed to the monokernel forward: {sorted(got)}, not {sorted(want)}"
            )
        widest = max(
            res["batches"].values(),
            key=lambda b: len(b["prompt"]),
            default=dict(prompt=[0]),
        )["prompt"]
        print(json.dumps(dict(
            role=res["role"], startup_s=res["startup_s"], asm_env=res["asm_env"], routed_line=res["routed_line"],
            prompts=res["prompts"], prompt_tokens=[min(widest), max(widest)],
            decode_rows=dict(Counter(res["decode_rows"])),
            cached_share={m: round(sum(c or 0 for c in b["cached"]) / sum(b["prompt"]), 3) for m, b in res["batches"].items()},
            wall_s={m: b["wall_s"] for m, b in res["batches"].items()},
        ), indent=1))  # fmt: skip
    print("PASS" if not fails else "FAIL: " + "; ".join(fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
