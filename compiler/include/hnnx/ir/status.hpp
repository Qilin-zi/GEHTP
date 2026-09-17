#pragma once
// ============================================================================
// GraphStatus —— 全局作用域 struct (SDK graph_status.h 原样), hnnx 内别名。
//
// M32 重写: 原三值枚举 (OK/FAIL/SKIP) 是 interface_defs.h 的旧 C API 枚举,
// 不是本族 .so 中真实存在的类型。SDK core 头全套 (op_io_ptrs.h/
// typical_op.h/op_hook_base.h/meta_op_base.h) 与 libHtpPrepare.so 反汇编
// 共同钉死: GraphStatus 是包裹 enum GraphErrorCode 的单成员 struct:
//
// 证据 (libHtpPrepare.so):
//   * OpIoPtrs ctor @0xf85070: movl $0xffffffff,0x18(%rdi)  —— stat 初值
//     是 -1 (ErrorFatal), 不是 {0,1,2} 三值枚举;
//   * OpIoPtrs ctor @0xf84a10: 失败路径 stat = ErrorFatal(-1), 成功路径
//     stat = 0, 且失败日志文案 "...ERROR:Can't find input tensor..." 与
//     GraphPrepare 报错族一致;
//   * TypicalOpIO<Ftype>::prepare 返回 ErrorNoTCM(5) (typical_op.h 注释,
//     WITH_PREPARE 路径), TypicalOpUtil::output_allocate 的 hook 否决值
//     直接透传 —— 非 {0,1,2} 可表示;
//   * OpIoPtrs/OpHookBase 全部方法签名 `GraphStatus (…)(OpIoPtrs const&,
//     Op&) const`, pmf 首字 {1,0}/{9,0} 对应 vtable 偏移, 返回值按 int
//     在 eax 传递 (单成员 enum 底衬 int, struct 非 POD-trivial 但同尺寸
//     4 字节, 寄存器返回)。
//
// 值域以 SDK graph_status.h 枚举为准 (op_io_ptrs.h 失败路径用 ErrorFatal,
// ophook 语义用 NotApplicable 作"无意见"哨兵)。
// ============================================================================

struct GraphStatus {
    enum GraphErrorCode {
        Success = 0,
        ErrorPickleSkipped = 1,
        ErrorDimensions = 2,
        ErrorPrecision = 3,
        ErrorNAN = 4,
        ErrorNoTCM = 5,
        ErrorNoSpace = 6,
        ErrorUnsupported = 7,
        ErrorSequence = 8, // e.g. adding a node after prepare
        ErrorBadID = 9, // source ref was 0 or not defined in graph; node ID was 0 or duplicate.
        ErrorBadInput = 10,
        ErrorInvalidTCM = 11,
        ErrorFatalSchdule = 12,
        ErrorFatalTCMRequest = 13,
        ErrorFatalAllocate = 14,
        ErrorFatalCheck = 15, // preprocess in prepare, e.g. clear the opid_alias_map, check connectivity, order_nodes
        ErrorBadOpName = 16,
        ErrorFatalOptimize = 17,
        ErrorFatalCSE = 18, // steps that combined with CSE e.g. dead_code_removal_and_cse, const_prop_and_cse()
        ErrorFatalInsert = 19, // when inject DMA spill/fill to fix any oversubscription of TCM
        ErrorFatalReschedule = 20,
        ErrorEmptyList = 21,
        ErrorFatalExecute = 22,
        ErrorFatalExecuteLastRun = 23,
        ErrorTCMAcquire = 24, // we can recover from TCM acquire failures (when tcm was locked by a different client)
        ErrorHMXAcquire = 25,
        ErrorHMXPower = 26,
        ErrorBadPMU = 27,
        ErrorThreadCounts = 28,
        ErrorClobberedPMU = 29, // Something clobbered our expected PMU event.
        ErrorWeightsCompressedNoAperture = 30, // Weights are DLBC compressed, but failed to acquire aperture for it
        ErrorRank = 31,
        ErrorHMXRelease = 32,
        ErrorTCMRelease = 33,
        ErrorWeightsCompressedBadFormat = 34, // Weights are DLBC compressed, but compression format is not supported
        ErrorTCMReleaseReacquire = 35,
        ErrorFatalCheckpoint = 36,

        ErrorFatalMcMetaData = 93,
        ErrorFatalApiRecVersion = 94,
        ErrorFatalDeserialize = 95,
        ErrorFatalBlobVersion = 96,
        ErrorFatalBlobVtcmSize = 97,
        ErrorFatalUnusableGraph = 98,
        ErrorFatalException = 99,
        NotApplicable = 100, // used for internal signaling, should not be returned from API
        Yielding = 101,
        AbortSuccess = 102,
        ErrorBadDynamicOp = 103,
        ErrorNot2kAlignedVTCMReq = 104,
        ErrorBadCDPatchContent = 105,
        ErrorSegmentMemoryOverflow = 106,
        ErrorBadAlloc = 107,
        ErrorBadCDExtraSize = 108,
        ErrorInsufficientCDExtraSize = 109,
        ErrorFatal = -1,
    };

    GraphStatus(const GraphStatus &) = default;
    GraphStatus &operator=(const GraphStatus &) = default;
    GraphStatus(GraphErrorCode ec) : error_code(ec) {}
    explicit GraphStatus(int ec) : error_code(static_cast<GraphErrorCode>(ec)) {}
    int to_int() const { return static_cast<int>(error_code); }
    operator bool() const { return error_code != Success; }

    bool operator==(GraphErrorCode ec) const { return error_code == ec; }
    bool operator!=(GraphErrorCode ec) const { return error_code != ec; }

  private:
    GraphErrorCode error_code;
};

// ir 族头 (op.hpp 等) 历史上在 hnnx 命名空间内引用 GraphStatus —— 保持
// `using hnnx::GraphStatus;` 继续可用 (SDK 侧同样如此: 全局定义 + 使用点
// 多在 hnnx 内)。
namespace hnnx {
using ::GraphStatus;
} // namespace hnnx
