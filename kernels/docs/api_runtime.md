# Runtime API 参考

声明: [`include/hvxhmx_runtime.h`](../include/hvxhmx_runtime.h)

所有 HMX kernel 前必须初始化 runtime. 这是库使用的第一件事.

## 生命周期总览

```
hmx_runtime_setup(2MB)          ← 程序开始, 调一次 (幂等)
      │
      ├── hmx_power_on()              (setup 内部已调, 通常不用手动)
      ├── 申请 VTCM + memset 清零      (防 CX_FAULT)
      │
      ├── ... 调各种 kernel ...
      │       hmx_convf16 / hvhx_divide_u8 / ...
      │
      └── hmx_perf_now_us()           (可选, 计时)
      
hmx_runtime_teardown()          ← 程序退出, 释放 VTCM + 下电
```

## hmx_runtime_setup

```c
int hmx_runtime_setup(unsigned int vtcm_size);
```

初始化 HMX 运行时. 内部依次:
1. `hmx_power_on()` — DCVS_v3 PERFORMANCE + HVX/HMX power_up
2. 申请 `vtcm_size` 字节 VTCM (2KB 对齐)
3. memset 清零 VTCM (防 CX_FAULT)

| 参数 | 说明 |
|------|------|
| `vtcm_size` | 申请的 VTCM 字节数, 必须 2KB 对齐. 建议 `2*1024*1024`. |

返回: 0 成功, 非 0 HAP 错误码.

幂等: 重复调用安全 (内部检测已初始化). 任何 HMX kernel 调用前必须先 setup.

```c
if (hmx_runtime_setup(2 * 1024 * 1024) != 0) {
    /* CDSP/fastrpc 未就绪, 或 VTCM 不足 */
}
```

## hmx_runtime_teardown

```c
void hmx_runtime_teardown(void);
```

归还 VTCM + HMX 下电. 与 setup 配对. 程序退出前调.

## hmx_power_on

```c
int hmx_power_on(void);
```

DCVS_v3 PERFORMANCE + HVX/HMX power_up. 返回 0 成功.

> 通常无需手动调 — `hmx_runtime_setup` 内部已调. 仅在手动管理电源时使用.

## 执行控制

```c
void hmx_enable_execution(void);    /* 使能本线程 HMX 执行 */
void hmx_disable_execution(void);   /* 关闭 */
void hmx_unit_acquire(void);        /* 独占获取 HMX 单元 */
void hmx_unit_release(void);        /* 释放 */
```

setup 内部已调 enable + acquire. 调用方一般不用碰, 除非需要多线程串行化访问 HMX.

## VTCM 查询

```c
void        *hmx_runtime_get_vtcm_base(void);   /* VTCM 基地址 (setup 后有效) */
unsigned int hmx_runtime_get_vtcm_size(void);   /* 字节数 */
unsigned int hmx_runtime_get_ctx_id(void);      /* HMX context id */
```

高级用户直接操作 VTCM (手工打包 crouton 给低级 core) 时用. 一般高层 API 内部处理.

## 计时

```c
long long hmx_perf_now_us(void);
```

返回当前 qtimer 微秒数 (HAP_perf). 用于 kernel 性能测量:

```c
long long t0 = hmx_perf_now_us();
for (int i = 0; i < 500; i++) hmx_convf16(act, wgt, bias, out, 32, 32, 32);
long long t1 = hmx_perf_now_us();
double avg_us = (double)(t1 - t0) / 500;
```

## 便利内联

```c
static inline int          hmx_k_aligned(uint32_t K);            /* K%32==0? */
static inline unsigned int hmx_dW_limit(uint32_t n_tiles, size_t tile_size);
```

`hmx_k_aligned` 校验 K 是否 32 倍数 (HMX crouton 边界要求). `hmx_dW_limit` 算链式
tile 加载的 dW 限值 (高级).

## 常量

```c
#define HMX_TILE_DIM      32       /* 单 tile 32×32 */
#define HMX_FP16_TILE_SZ  2048     /* fp16 crouton 2KB */
#define HMX_U8_TILE_SZ    1024     /* u8 crouton 1KB */
#define HMX_VTCM_ALIGN    2048     /* VTCM 2KB 对齐要求 */
```

调用方用来给 buffer 取尺寸/对齐.

## 故障

| 现象 | 原因 | 处理 |
|------|------|------|
| setup 返回非 0 | CDSP/fastrpc 未就绪 | host 侧重连 (见 USERGUIDE §故障排查) |
| kernel 跑完结果全 0 / bias | VTCM 未清零 (CX_FAULT) | 确认 setup 内部 memset 执行 |
| CDSP crash (ERR_FATAL) | vgather 指令 / VLA 栈溢出 | build 检查 vgather=0; 用 static buffer |
| 多次 setup 内存涨 | 未 teardown | 程序退出前 teardown |
