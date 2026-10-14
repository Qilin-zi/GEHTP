# GEHTP 专有名词小词典

> 设备侧/编译器术语一页速查。按主题分组，只讲直觉不讲教科书。
> 汇报里碰到看不懂的词，先来这里查；缺的词随时补进来。

## 1. 芯片与执行环境

- **Hexagon (Q6)** — 高通的 DSP 处理器核，我们所有设备代码跑在上面。
- **CDSP** — Consumer DSP，Hexagon 里管通用计算的那个域；我们的 runner/lib 都加载到 domain 3。
- **PD** — DSP 上的"进程"。unsigned PD = 未签名沙箱，开发期用这个（正式交付才走签名 lib）。
- **fastrpc** — host（ARM 侧）调用 DSP 函数的 RPC 机制。`run_main_on_hexagon` 就是用 fastrpc 把 runner.so 的 main() 调到 DSP 上执行。坑：fastrpc 按**文件名**缓存已加载的 lib，同名重推不生效，每次换新名字。
- **HVX** — Hexagon 向量单元（128B/向量）。擅长"对一排数做同样的事"：elementwise、归约、纯搬运。
- **HMX** — Hexagon 矩阵乘单元。f16 GEMM / conv / W4A16 最终都落到它。硬约束：**只能读 VTCM，不能直接读 DDR**。
- **VTCM** — 紧贴计算单元的片上内存，实测 16MB，带宽极高。想成"操作台面"：小而快，HMX 只能从台面上取数。
- **DDR** — 主内存（GB 级）。大而慢，想成"仓库"。
- **板 / 器件** — 跑真机的开发板（序列号 d0f1784），多会话共用，所以推上去的所有文件都要带私有名。

## 2. blob 与编译器侧

- **blob (.wtop)** — 编译器吐出的"执行清单"：槽表 + op 列表 + 权重数据 + 若干 footer。执行器照单顺序执行，不做任何图分析。
- **emit** — 编译器生成 blob 的动作/代码（wtop_emit.cpp）。
- **slot** — blob 里编号的数据块（权重/输入/各种标记），op 用槽号引用。
- **temp** — 运行时中间结果缓冲，按 id 编号。**TEMPOFF 槽** = blob 里那段"临时区规划表"，写明每个 temp 放 DDR 池还是 VTCM 池、偏移多少。
- **opcode / arity** — 算子编号 / 参数个数。例：OP_MATMUL_F16 = 0x18，arity 7。
- **footer** — blob 尾部按 magic 探测的可选元数据段：v3 引擎注解、carve footer、GDN 几何表、VTCL 布局表。
- **位级 / bit-exact** — 输出逐字节相同。比"数值近似相等"硬得多，是"重构没改变行为"的最强证据。
- **双门** — 精度门 + 性能门，两把尺都要过。

## 3. VTCM 三区与布局（013 A 段）

- **布局表 (VTCL)** — 新加的 32 字节小表，贴在 TEMPOFF 表尾，写明 VTCM 16MB 怎么切三区。执行器**装载时读一次**，之后不再现场决定。
- **pin 区** — VTCM 开头一段，放"钉住不放"的热权重。
- **arena 区** — f16 matmul / conv 的 HMX 专用工作区（激活/权重/输出的打包面都放这）。
- **resid（驻留）区** — VTCM 尾部，放编译器决定"整个 run 住进 VTCM"的 temp。
- **驻留** — 某数据整个 run 期间留在 VTCM，不反复搬进搬出。
- **三区互斥** — pin/arena/resid 互不重叠，装载期硬校验，漂移即报错。

## 4. 运行时机制

- **wtcache** — 设备侧"VTCM 总管家"模块。`wtcache_open` = 一次性把整块 VTCM 拿到手 + 开 HVX/HMX 电源。谁想用 VTCM 都先经过它。
- **carve** — W4A16 矩阵乘开工前"布置台面"：按 M/K/N 在 arena 里划出 act/wt/bias 等小份区域（dc_w4_carve）。
- **re-carve** — 下一个 matmul 形状不同时要重新划区。附带动作：旧路径每次 re-carve 都 flush 整个 16MB VTCM；013 改成只 flush arena 段。
- **flush / invalidate** — CPU cache 一致性动作。**flush** = 把 CPU cache 里改过的内容写回内存（让别人看得见新数据）；**invalidate** = 作废 cache 里的旧副本（让自己读到别人的新写）。CPU / DMA / HMX 三方看同一块内存时，漏做就会读旧数据——设备侧数值错乱的头号来源。
- **探戈** — HMX runtime 和 wtcache 都想独占整块 VTCM；旧代码里后来者把前者拆掉自己上，来回拆装像跳探戈。013 A 段目标：装载期一次分区，谁都不许再拆，tango 计数器 = 0 即证据。
- **attach / setup_region** — 不拆别人的，只把自己的工作区登记到台面某段：`hmx_runtime_setup_region(arena基址, arena大小)`。013 的有界区域方案。
- **懒初始化** — 旧路径不提前开 wtcache，等第一个需要 VTCM 的 op 到了才开（并可能触发探戈）。
- **bump 池** — 线性递增分配的内存池（指针只前进不回收），省掉成百上千次 malloc/free。

## 5. 数值与算子

- **f16 (fp16)** — 半精度浮点，2 字节。0.8B 主线数值格式；纪律 = f32 累加、f16 存储。
- **W4A16** — 权重 4bit、激活 16bit 的量化矩阵乘（权重 0.5 字节/元素）。
- **Q4_0** — GGUF 的 4bit 分块量化格式（每 32 元素配一个 f16 scale）。lm_head/embed 用它。
- **crouton** — HMX 消费的 32×32 tile 打包布局（pair-interleave）。数据进 HMX 前必须重排成它。
- **GEMM / GEMV** — 矩阵×矩阵 / 矩阵×向量。M<32 是 GEMV，走 HVX。
- **dequant** — 反量化：4bit 整数 × scale → f16。

## 6. 验证与取证

- **runner (42_gehtp_runner)** — 设备侧小程序：读 job.txt → 调执行器 → 结果写回文件。
- **job.txt** — runner 的任务单（blob=、input=、output=、out_temp= 等键）。
- **FARF** — DSP 上的 printf。走 diag 通道，**不一定**能在 logcat 看到；重要取证改写 optrace 文件。
- **optrace** — /data/local/tmp/hrt/gehtp/optrace.txt，执行器逐 op / 里程碑的文件日志，比 FARF 可靠。
- **op_ts** — 逐 op 时间戳（start/dur/dma/engine/planned/deviation），性能归因数据源。
- **golden / cos 门 / top1** — 金标输出；余弦相似度门槛（如 0.999930）；top1 = 最大值位置与金标一致数（32/32 = 全对）。
- **双跑对拍** — 同一 blob 用新旧 lib 各跑一遍比输出（位级或 cos）。
- **私有名纪律** — 板子多人共用，推上去的 lib/blob/job/结果全带会话标签（如 016b）防互相覆盖。

## 7. 流程用语

- **锚点** — 公认的基线数字（gdn cos 0.999930 / exec 2.76s），改完不许变差。
- **零回归** — 改动后锚点全部保持。
- **主线** — origin/task/prof-wp 分支，回合结束 merge 回去。
- **合成 blob** — 手工构造的小 blob（几个 op），专门用来验证某条机制路径，不等大模型编译。
