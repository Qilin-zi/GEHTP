# test_assets — GEHTP 测试资产暂存

门户（gehtp compile+run）与各窗口共享的设备验证资产。`/tmp` 有高频清理（一日两清实锤），凡是要活过重启的资产一律放这里。

## 布局

| 目录 | 内容 | 来源 | 判据 |
|---|---|---|---|
| `l3/` | layer_3.wtop(45MB, 81 op 注意力层)+ in/gold.f16.raw + manifest | 设备 `/data/local/tmp/hvxhmx23/g40/`(M4.2 资产,兜底源) | `gehtp run` 输出 vs gold，**值差判据**（期望 max_valdiff≈0.000488;81-op f16 链禁位判据） |

## 纪律

- `*.wtop` / `*.raw` 已被根 .gitignore 覆盖，不会进 git;manifest/README 可跟踪
- 设备 g40/ 是 L3 资产的兜底源，本地丢失可重拉：
  `adb -s 52f67807 pull /data/local/tmp/hvxhmx23/g40/{layer_3.wtop,in.f16.raw,gold.f16.raw} .`
- disk2-ca 例41 收口后会将 test_models 的 layer_0/layer_3 更新为 16af35d 新编码（rank 三修复）；本目录 l3/ 是 portal 回归快照，届时同步刷新
- L0(903 op, 119MB）资产由例41线（disk2-ca）持有，收口后迁入 `l0/`

## 快速复跑（设备互斥：先 ps 查 run_main_on_hexagon)

```bash
scripts/gehtp run test_assets/l3/layer_3.wtop \
    --input test_assets/l3/in.f16.raw --output /tmp/l3_out.f16.raw
```
