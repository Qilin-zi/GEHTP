# tests/ — 测试

## test_ir.cpp — 单元测试 (10 项)

验证基础组件:
1. Fibonacci hash
2. OpDef_Const 构造
3. 6 阶段 PHASE 阈值 (3000/10190/11892/12492/21101/22000)
4. Serializer tag 编码
5. CostSource init
6. SequencerConfig
7. VtcmCacheInstance
8. TilingRegistry + SimpleTiler
9. McastOptimizer
10. OpEmitter

## test_e2e.cpp — 端到端 (21 步)

完整管线验证:
1-6. 建图: Input -> Conv -> Relu -> Softmax -> Output (含权重 const 节点)
6b. **fusion 验证**: Conv+Relu 融合成 ConvActivations
7. 优化 pass
8. 序列化 (5104 字节, 含真实权重)
9. cost model
10. 量化往返 (int8)
11. weight scatter
12. tiling (2x2)
13. VTCM
14. multicast supercast
15. DP sequencer 排序
16. .bin 头部 (0xe3471cb8 = TAG_IO_TENSORS_CONFIG)
16b. **反序列化往返**: 重建图, ConvActivations inputs=[5,1], const offset=4 size=3456
16c. **再序列化确定性**: 1600 字节
17. 154 op 注册
18. op 工厂生成 Conv
19. **execute**: Relu([-1,2,-3,...])->[0,2,0,4,0,6,0,8], Add([1,2,3,4]+[10,20,30,40])->[11,22,33,44]

## 运行

```bash
$env:PATH = "<mingw64\bin>;" + $env:PATH
build/test_ir.exe   # exit 0 = All tests passed
build/test_e2e.exe  # exit 0 = End-to-End PASSED
```
