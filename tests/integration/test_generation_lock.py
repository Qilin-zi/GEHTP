"""并发回归: persistent 后端 KV/SSM 是设备全局单状态 (reset 不分 handle),
并发请求的设备步进必须整代互斥, 否则两代 token 交错写进同一份 KV。"""

import threading
import time

import gehtp_server


def test_concurrent_generations_do_not_interleave(client, registered_model, monkeypatch):
    runtime = gehtp_server.get_runtime()
    orig_run = runtime.run
    calls = []

    def slow_run(handle, input_ids, position, kv_position):
        calls.append(position)
        time.sleep(0.01)
        return orig_run(handle, input_ids, position, kv_position)

    monkeypatch.setattr(runtime, "run", slow_run)

    payload = {
        "model": registered_model,
        "messages": [{"role": "user", "content": "hello world"}],
        "max_tokens": 2,
    }

    barrier = threading.Barrier(2)
    results = {}

    def hit(name):
        barrier.wait(timeout=10)
        results[name] = client.post("/v1/chat/completions", json=payload)

    threads = [threading.Thread(target=hit, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads)

    assert len(results) == 2
    for name, r in results.items():
        assert r.status_code == 200, (name, r.text)
        assert r.json()["choices"][0]["finish_reason"] == "length"

    starts = [i for i, p in enumerate(calls) if p == 0]
    assert len(starts) == 2, f"应恰有两代 (各起于 position 0): {calls}"
    mid = starts[1]
    assert calls[:mid] == list(range(mid)), f"第一代步进被交错: {calls}"
    assert calls[mid:] == list(range(len(calls) - mid)), f"第二代步进被交错: {calls}"
