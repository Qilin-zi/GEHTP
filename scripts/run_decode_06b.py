#!/usr/bin/env python3
"""run_decode_06b.py — Qwen3-0.6B 设备端自回归贪心 decode 验证.

与 HF golden (gen_decode_golden_06b.py) 对拍: 逐 token 喂 position, 设备
argmax 贪心, 断言 generated_ids == golden.generated_ids.

0.6B face 几何 (vs 0.8B): ROT_DIM=128 (full rotary, head_dim=128),
rope_theta=1e6, FACE_N32=160 (cos128+sin128+mask64 u16), IN_BYTES=644。
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/disk1/gehtp/scripts")
import persistent_runtime as pr  # noqa: E402

# ---- 0.6B face 几何 ----
VOCAB = 151936
MAX_POS = 64
ROT_DIM = 128
FACE_N32 = 160
IN_BYTES = 4 + 4 * FACE_N32            # 644
_INV = 1.0 / (1e6 ** (np.arange(0, 64, dtype=np.float64) / 64))  # half=64


def make_face(position: int) -> bytes:
    """0.6B decode 融合输入面 (640B): [cos 128 | sin 128 | mask 64] u16 → i32."""
    if not 0 <= position < MAX_POS:
        raise ValueError(f"position {position} out of [0,{MAX_POS})")
    h = np.zeros(2 * FACE_N32, dtype=np.uint16)          # 320 u16
    c = np.cos(position * _INV)                          # 64
    s = np.sin(position * _INV)                          # 64
    h[0:128] = np.concatenate([c, c]).astype(np.float16).view(np.uint16)   # cos
    h[128:256] = np.concatenate([s, s]).astype(np.float16).view(np.uint16)  # sin
    m = np.zeros(MAX_POS)
    m[position + 1:] = -np.inf
    h[256:320] = m.astype(np.float16).view(np.uint16)    # causal mask
    return h.view(np.int32).tobytes()


def make_step_input(token_id: int, position: int) -> bytes:
    """单步请求体: ids i32[1] + face 640B = 644B."""
    return np.array([token_id], dtype=np.int32).tobytes() + make_face(position)


# monkeypatch 模块级 (run() 在调用点解析模块全局)
pr.MAX_POS = MAX_POS
pr.ROT_DIM = ROT_DIM
pr.FACE_N32 = FACE_N32
pr.IN_BYTES = IN_BYTES
pr._INV = _INV
pr.make_face = make_face
pr.make_step_input = make_step_input


def _server_running_q6b(self) -> bool:
    r = self._adb("shell", "pgrep -f '[g]ehtp_persistent_q6b.so' || true")
    return bool(r.stdout.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="d0f1784")
    ap.add_argument("--blob", default="/data/local/tmp/hvxhmx23/gehtp_q6b/decode.wtop")
    ap.add_argument("--out-temp", type=int, required=True)
    ap.add_argument("--golden",
                    default="/disk1/gehtp/test_models/qwen3_06b/decode/golden_tokens.json")
    ap.add_argument("--max-new", type=int, default=32)
    args = ap.parse_args()

    golden = json.load(open(args.golden))
    prompt_ids = golden["prompt_ids"]
    want = golden["generated_ids"][: args.max_new]

    rt = pr.PersistentDeviceRuntime(device=args.device)
    rt.PERSIST_DIR = "/data/local/tmp/hvxhmx23/gehtp_persist_q6b"
    pr.PersistentDeviceRuntime.server_running = _server_running_q6b

    rt.start_server(blob=args.blob, out_temp=args.out_temp, in_bytes=IN_BYTES,
                    so_name="gehtp_persistent_q6b.so", idle_exit_s=1800)
    h = rt.init("qwen3_06b_decode", kv_max=MAX_POS, wtop_path=args.blob)
    rt.reset(h)

    position = 0
    for tok in prompt_ids:                       # prefill: 建 KV
        nxt = rt.run(h, [tok], position, position)
        position += 1
    generated = [nxt]                            # 末 prompt token 输出 = 首生成 token
    for _ in range(1, args.max_new):             # 生成相位
        nxt = rt.run(h, [nxt], position, position)
        position += 1
        generated.append(nxt)

    print(f"[device] prompt_ids={prompt_ids}")
    print(f"[device] generated={generated}")
    print(f"[golden] generated={want}")
    n_match = sum(a == b for a, b in zip(generated, want))
    print(f"[match] {n_match}/{args.max_new}")
    if generated == want:
        print("E2E DECODE PASS: device greedy == HF greedy")
        return 0
    for i, (a, b) in enumerate(zip(generated, want)):
        if a != b:
            print(f"  first divergence @ step {i}: device={a} golden={b}")
            break
    print("E2E DECODE MISMATCH")
    return 1


if __name__ == "__main__":
    sys.exit(main())
