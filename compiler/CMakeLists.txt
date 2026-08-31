cmake_minimum_required(VERSION 3.16)
project(htp_reimpl CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W3 /bigobj /Zc:__cplusplus)
else()
    add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-unused-but-set-variable)
endif()

# Main library
add_library(htp_core STATIC
    src/ir/opdef.cpp
    src/ir/graph_prepare.cpp
    src/ir/graph_deps.cpp
    src/ir/graph_dumper.cpp
    src/ir/tensor.cpp
    src/ir/op_registry.cpp
    src/opt/optimization_passes.cpp
    src/opt/fixpoint.cpp
    src/cost/cost_model.cpp
    src/vtcm/fancy_allocator.cpp
    src/vtcm/cp_solver.cpp
    src/vtcm/dp_group_graph.cpp
    src/vtcm/supertile.cpp
    src/tiling/tiler.cpp
    src/tiling/conv_tiling.cpp
    src/serialize/serializer.cpp
    src/serialize/deserializer.cpp
    src/serialize/bin_format.cpp
    src/serialize/barrel.cpp
    src/serialize/class_index_record.cpp
    src/serialize/co_aux_record.cpp
    src/serialize/context_binary_writer.cpp
    src/schedule/scheduler.cpp
    src/schedule/e2e_bridge.cpp
    src/serialize/deserz_except.cpp
    src/serialize/deserz_shared_obj.cpp
    src/serialize/serializer_bridge.cpp
    src/tens/crate_records.cpp
    src/tens/interface_table.cpp
    src/tens/shape.cpp
    src/vtcm/alloc_io_tensors_m38.cpp
    src/vtcm/create_supertiles_m36.cpp
    src/vtcm/phys_alloc_m37.cpp
    src/vtcm/spillfill_g4.cpp
    src/vtcm/tcm_migration_m42.cpp
    src/vtcm/tile_shape_m36c.cpp
    src/dma/spill_fill.cpp
    src/dma/op_emitter.cpp
    src/mcast/mcast_optimizer.cpp
    src/scheduler/dp_sequencer.cpp
    src/scheduler/mlh_model.cpp
    src/scheduler/st_cut.cpp
    src/api/c_interface.cpp
    src/api/hexagon_nn_env.cpp
    src/frontend/qnn_ir_loader.cpp
    src/ops/ops.cpp
    src/ops/hlx_hmx.cpp
    src/ops/quantize.cpp
    src/ops/weights.cpp
)

target_include_directories(htp_core PUBLIC include)
# 测试数据路径便携化(见 tests/test_paths.hpp): Linux/Windows 双路径
target_compile_definitions(htp_core PUBLIC
    HNNX_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test_models"
    HNNX_GEHTP_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../test_models"
    HNNX_REF_DIR="${CMAKE_CURRENT_SOURCE_DIR}/reference")

# Tests
enable_testing()
add_executable(test_ir tests/test_ir.cpp)
target_link_libraries(test_ir htp_core)
add_test(NAME test_ir COMMAND test_ir)

add_executable(test_e2e tests/test_e2e.cpp)
target_link_libraries(test_e2e htp_core)
add_test(NAME test_e2e COMMAND test_e2e)

add_executable(test_execute_chain tests/test_execute_chain.cpp)
target_link_libraries(test_execute_chain htp_core)
add_test(NAME test_execute_chain COMMAND test_execute_chain)

add_executable(test_matmul_add tests/test_matmul_add.cpp)
target_link_libraries(test_matmul_add htp_core)
add_test(NAME test_matmul_add COMMAND test_matmul_add)

add_executable(test_context_bin tests/test_context_bin.cpp)
target_link_libraries(test_context_bin htp_core)
add_test(NAME test_context_bin COMMAND test_context_bin)

add_executable(test_qnn_ir_loader tests/test_qnn_ir_loader.cpp)
target_link_libraries(test_qnn_ir_loader htp_core)
add_test(NAME test_qnn_ir_loader COMMAND test_qnn_ir_loader)

add_executable(test_transformer2 tests/test_transformer2.cpp)
target_link_libraries(test_transformer2 htp_core)
add_test(NAME test_transformer2 COMMAND test_transformer2)

add_executable(test_transformer2_before tests/test_transformer2_before.cpp)
target_link_libraries(test_transformer2_before htp_core)
add_test(NAME test_transformer2_before COMMAND test_transformer2_before)

add_executable(test_before_after_dump tests/test_before_after_dump.cpp)
target_link_libraries(test_before_after_dump htp_core)
add_test(NAME test_before_after_dump COMMAND test_before_after_dump)

add_executable(test_general_scheduler tests/test_general_scheduler.cpp)
target_link_libraries(test_general_scheduler htp_core)
add_test(NAME test_general_scheduler COMMAND test_general_scheduler)

add_executable(test_multi_model tests/test_multi_model.cpp)
target_link_libraries(test_multi_model htp_core)
add_test(NAME test_multi_model COMMAND test_multi_model)

add_executable(test_multi_model_write tests/test_multi_model_write.cpp)
target_link_libraries(test_multi_model_write htp_core)
add_test(NAME test_multi_model_write COMMAND test_multi_model_write)

add_executable(test_typical_structures tests/test_typical_structures.cpp)
target_link_libraries(test_typical_structures htp_core)
add_test(NAME test_typical_structures COMMAND test_typical_structures)

add_executable(test_graph_parse_consistency tests/test_graph_parse_consistency.cpp)
target_link_libraries(test_graph_parse_consistency htp_core)
add_test(NAME test_graph_parse_consistency COMMAND test_graph_parse_consistency)

add_executable(test_before_graph_strict tests/test_before_graph_strict.cpp)
target_link_libraries(test_before_graph_strict htp_core)
add_test(NAME test_before_graph_strict COMMAND test_before_graph_strict)

add_executable(test_full_graph_match tests/test_full_graph_match.cpp)
target_link_libraries(test_full_graph_match htp_core)
add_test(NAME test_full_graph_match COMMAND test_full_graph_match)

add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops htp_core)
add_test(NAME test_ops COMMAND test_ops)

add_executable(test_vtcm_reuse tests/test_vtcm_reuse.cpp)
target_link_libraries(test_vtcm_reuse htp_core)
add_test(NAME test_vtcm_reuse COMMAND test_vtcm_reuse)

add_executable(test_cp_solver tests/test_cp_solver.cpp)
target_link_libraries(test_cp_solver htp_core)
add_test(NAME test_cp_solver COMMAND test_cp_solver)

add_executable(test_cp_e2e tests/test_cp_e2e.cpp)
target_link_libraries(test_cp_e2e htp_core)
add_test(NAME test_cp_e2e COMMAND test_cp_e2e)

add_executable(test_dma_synctoken tests/test_dma_synctoken.cpp)
target_link_libraries(test_dma_synctoken htp_core)
add_test(NAME test_dma_synctoken COMMAND test_dma_synctoken)

add_executable(test_mcast_ilp tests/test_mcast_ilp.cpp)
target_link_libraries(test_mcast_ilp htp_core)
add_test(NAME test_mcast_ilp COMMAND test_mcast_ilp)

add_executable(test_cost_model tests/test_cost_model.cpp)
target_link_libraries(test_cost_model htp_core)
add_test(NAME test_cost_model COMMAND test_cost_model)

add_executable(test_phase4_integration tests/test_phase4_integration.cpp)
target_link_libraries(test_phase4_integration htp_core)
add_test(NAME test_phase4_integration COMMAND test_phase4_integration)

add_executable(test_transformer2_prepare tests/test_transformer2_prepare.cpp)
target_link_libraries(test_transformer2_prepare htp_core)
add_test(NAME test_transformer2_prepare COMMAND test_transformer2_prepare)

add_executable(test_simple_linear_prepare tests/test_simple_linear_prepare.cpp)
target_link_libraries(test_simple_linear_prepare htp_core)
add_test(NAME test_simple_linear_prepare COMMAND test_simple_linear_prepare)

add_executable(test_simple_linear_pipeline tests/test_simple_linear_pipeline.cpp)
target_link_libraries(test_simple_linear_pipeline htp_core)
add_test(NAME test_simple_linear_pipeline COMMAND test_simple_linear_pipeline)

add_executable(test_st_cut tests/test_st_cut.cpp)
target_link_libraries(test_st_cut htp_core)
add_test(NAME test_st_cut COMMAND test_st_cut)

add_executable(test_m36_supertile tests/test_m36_supertile.cpp)
target_link_libraries(test_m36_supertile htp_core)
add_test(NAME test_m36_supertile COMMAND test_m36_supertile)

add_executable(test_small_graph_e2e tests/test_small_graph_e2e.cpp)
target_link_libraries(test_small_graph_e2e htp_core)
add_test(NAME test_small_graph_e2e COMMAND test_small_graph_e2e)

# M33-M42 新模块测试(迁移后补注册,阶段 4-7 的回归基)
add_executable(test_m33_serializer tests/test_m33_serializer.cpp)
target_link_libraries(test_m33_serializer htp_core)
add_test(NAME test_m33_serializer COMMAND test_m33_serializer)

add_executable(test_m34_shape tests/test_m34_shape.cpp)
target_link_libraries(test_m34_shape htp_core)
add_test(NAME test_m34_shape COMMAND test_m34_shape)

add_executable(test_m36c_tile_shape tests/test_m36c_tile_shape.cpp)
target_link_libraries(test_m36c_tile_shape htp_core)
add_test(NAME test_m36c_tile_shape COMMAND test_m36c_tile_shape)

add_executable(test_m37_phys_alloc tests/test_m37_phys_alloc.cpp)
target_link_libraries(test_m37_phys_alloc htp_core)
add_test(NAME test_m37_phys_alloc COMMAND test_m37_phys_alloc)

add_executable(test_m38_alloc_io tests/test_m38_alloc_io.cpp)
target_link_libraries(test_m38_alloc_io htp_core)
add_test(NAME test_m38_alloc_io COMMAND test_m38_alloc_io)

add_executable(test_m42_tcm_migration tests/test_m42_tcm_migration.cpp)
target_link_libraries(test_m42_tcm_migration htp_core)
add_test(NAME test_m42_tcm_migration COMMAND test_m42_tcm_migration)

# CLI: 图编译入口(阶段 8 扩展 --format tagged)
add_executable(hnnx_compile tools/hnnx_compile.cpp)
target_link_libraries(hnnx_compile htp_core)

# 阶段3: conv2d+add op 注册 + host 参考执行对拍
add_executable(test_conv_add_host tests/test_conv_add_host.cpp)
target_link_libraries(test_conv_add_host htp_core)
add_test(NAME test_conv_add_host COMMAND test_conv_add_host)

# 阶段4: const 池验证 + serialize_opdef 参数 schema + round-trip
add_executable(test_conv_add_bin tests/test_conv_add_bin.cpp)
target_link_libraries(test_conv_add_bin htp_core)
add_test(NAME test_conv_add_bin COMMAND test_conv_add_bin)

# 阶段5: Scheduler 接 ST-Cut(计划序 == 序列化序)
add_executable(test_scheduler_integration tests/test_scheduler_integration.cpp)
target_link_libraries(test_scheduler_integration htp_core)
add_test(NAME test_scheduler_integration COMMAND test_scheduler_integration)

# 阶段6: 单算子 conv 分块(公式级 + 序列化 + 数值等价)
add_executable(test_conv_tiling tests/test_conv_tiling.cpp)
target_link_libraries(test_conv_tiling htp_core)
add_test(NAME test_conv_tiling COMMAND test_conv_tiling)
