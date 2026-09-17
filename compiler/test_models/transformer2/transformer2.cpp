/* COPYRIGHT HEADER GOES HERE: No CopyRight Header String Passed During Model Conversion */

/* Command Line used:
qnn-onnx-converter; act_bitwidth=8; act_quantizer=tf; act_quantizer_calibration=min-max; act_quantizer_schema=asymmetric; adjust_nms_features_dims=True; algorithms=[]; align_matmul_ranks=True; apply_masked_softmax=uncompressed; arch_checker=False; backend=None; batch=None; bias_bitwidth=8; calc_static_encodings=False; converter_op_package_lib=; copyright_file=None; custom_io=; custom_op_config_paths=None; debug=-1; defer_loading=False; define_symbol=None; disable_batchnorm_folding=False; disable_defer_loading=False; disable_node_validation=False; disable_qnn_op_config_validation=False; disable_relu_squashing=False; dry_run=None; dumpIR=False; dump_custom_io_config_template=; dump_encoding_json=False; dump_inferred_model=False; dump_ir=; dump_ir_optimizer_config_template=False; dump_optimization_pass_mode_config=False; dump_pass_trace_info=False; dump_qairt_io_config_yaml=; dump_qairt_quantizer_command=None; dump_value_info=False; enable_framework_trace=False; enable_match_gathernd=False; enable_match_topk=False; enable_per_row_quantized_bias=False; exclude_named_tensors=False; expand_gru_op_structure=True; expand_lstm_op_structure=False; expand_sparse_op_structure=False; export_format=cpp; extract_color_transform=True; float_bias_bitwidth=0; float_bias_bw=32; float_bitwidth=32; float_bw=32; float_fallback=False; force_prune_cast_ops=False; handle_gather_negative_indices=True; ignore_encodings=False; include_data_invariant_ops=False; inject_cast_for_gather=True; input_dim=[['input', '1,8,64']]; input_dtype=[]; input_encoding=[]; input_layout=[]; input_list=None; input_type=[]; ir_optimizer_config=; keep_disconnected_nodes=False; keep_int64_inputs=False; keep_quant_nodes=False; keep_weights_quantized=False; match_caffe_ssd_to_tf=True; model_version=None; multi_time_steps_gru=False; multi_time_steps_lstm=False; no_simplification=False; op_package_lib=; optimization_pass_mode=ir_optimizer_mainline; optimization_pass_mode_config=; out_names=['output']; overwrite_model_prefix=False; pack_4_bit_weights=False; package_name=None; packed_masked_softmax_inputs=[]; packed_max_seq=1; param_quantizer=None; param_quantizer_calibration=min-max; param_quantizer_schema=asymmetric; percentile_calibration_value=99.99; perform_axes_to_spatial_first_order=True; perform_layout_transformation=False; prepare_inputs_as_params=False; preprocess_roi_pool_inputs=True; preserve_io=[]; preserve_onnx_output_order=False; quantization_overrides=; quantizer_log=None; quantizer_log_level=LogLevel.NONE; restrict_quantization_steps=[]; squash_box_decoder=True; unroll_gru_time_steps=True; unroll_lstm_time_steps=True; use_aimet_quantizer=False; use_convert_quantization_nodes=False; use_dynamic_16_bit_weights=False; use_native_dtype=False; use_native_input_files=False; use_native_output_files=False; use_per_channel_quantization=False; use_per_row_quantization=False; use_quantize_v2=False; validate_models=False; weights_bitwidth=8
*/

#include "QnnOpDef.h"
#include "QnnModel.hpp"

// Flag to determine if Backend should do node validation for each opNode added
#define DO_GRAPH_NODE_VALIDATIONS 1

using namespace qnn_wrapper_api;
const __attribute__((visibility("default"))) char* QNN_SDK_VERSION = "qaisw-v2.43.0.260127150333_193827";
extern "C" {
static ModelError_t addTensor_input(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_input[] = {1, 64, 8};
  VALIDATE(model.addTensor("input", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "input",
                                 .type= QNN_TENSOR_TYPE_APP_WRITE,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 3,
                                 .dimensions=dimensions_input,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=nullptr,
                                                .dataSize=0}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode_input_ncf(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR input_ncf */
  uint32_t dimensions_input_ncf_perm[] = {3};
  uint32_t input_ncf_perm[] = {0, 2, 1};
  Qnn_Param_t params_input_ncf[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "input_ncf_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_input_ncf_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)input_ncf_perm,
                           .dataSize=12}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs_input_ncf[] = {
    "input"
  };
  uint32_t dimensions_input_ncf[] = {1, 8, 64};
  Qnn_Tensor_t outputs_input_ncf[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "input_ncf",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions_input_ncf,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "input_ncf", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params_input_ncf, // Node Params
                         1, // Num Node Params
                         inputs_input_ncf, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_input_ncf, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer1_attn_Transpose(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_attn_Transpose */
  uint32_t dimensions__layer1_attn_Transpose_perm[] = {3};
  uint32_t _layer1_attn_Transpose_perm[] = {2, 0, 1};
  Qnn_Param_t params__layer1_attn_Transpose[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Transpose_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__layer1_attn_Transpose_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_layer1_attn_Transpose_perm,
                           .dataSize=12}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__layer1_attn_Transpose[] = {
    "input"
  };
  uint32_t dimensions__layer1_attn_Transpose_output_0[] = {8, 1, 64};
  Qnn_Tensor_t outputs__layer1_attn_Transpose[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Transpose_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer1_attn_Transpose_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_attn_Transpose", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__layer1_attn_Transpose, // Node Params
                         1, // Num Node Params
                         inputs__layer1_attn_Transpose, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer1_attn_Transpose, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__MatMul_289(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__MatMul_289[] = {192, 64};
  VALIDATE(model.addTensor("onnx__MatMul_289", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__MatMul_289",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_onnx__MatMul_289,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__MatMul_289),
                                                .dataSize=BINLEN(onnx__MatMul_289)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__layer1_attn_MatMul(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_attn_MatMul */
  const char*  inputs__layer1_attn_MatMul[] = {
    "_layer1_attn_Transpose_output_0",
    "onnx__MatMul_289"
  };
  uint32_t dimensions__layer1_attn_MatMul_output_0_fc[] = {8, 192};
  Qnn_Tensor_t outputs__layer1_attn_MatMul[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_MatMul_output_0_fc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__layer1_attn_MatMul_output_0_fc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_attn_MatMul", // Node Name
                         "qti.aisw", // Package Name
                         "FullyConnected", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer1_attn_MatMul, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__layer1_attn_MatMul, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer1_attn_MatMul_post_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_attn_MatMul_post_reshape */
  const char*  inputs__layer1_attn_MatMul_post_reshape[] = {
    "_layer1_attn_MatMul_output_0_fc"
  };
  uint32_t dimensions__layer1_attn_Unsqueeze_1_output_0[] = {1, 8, 1, 3, 64};
  Qnn_Tensor_t outputs__layer1_attn_MatMul_post_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Unsqueeze_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__layer1_attn_Unsqueeze_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_attn_MatMul_post_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer1_attn_MatMul_post_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer1_attn_MatMul_post_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer1_attn_Transpose_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_attn_Transpose_1 */
  uint32_t dimensions__layer1_attn_Transpose_1_perm[] = {5};
  uint32_t _layer1_attn_Transpose_1_perm[] = {3, 1, 2, 0, 4};
  Qnn_Param_t params__layer1_attn_Transpose_1[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Transpose_1_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__layer1_attn_Transpose_1_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_layer1_attn_Transpose_1_perm,
                           .dataSize=20}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__layer1_attn_Transpose_1[] = {
    "_layer1_attn_Unsqueeze_1_output_0"
  };
  uint32_t dimensions__layer1_attn_Transpose_1_output_0[] = {3, 8, 1, 1, 64};
  Qnn_Tensor_t outputs__layer1_attn_Transpose_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Transpose_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__layer1_attn_Transpose_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_attn_Transpose_1", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__layer1_attn_Transpose_1, // Node Params
                         1, // Num Node Params
                         inputs__layer1_attn_Transpose_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer1_attn_Transpose_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer1_attn_Squeeze_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_attn_Squeeze_1 */
  const char*  inputs__layer1_attn_Squeeze_1[] = {
    "_layer1_attn_Transpose_1_output_0"
  };
  uint32_t dimensions__layer1_attn_Squeeze_1_output_0[] = {3, 8, 1, 64};
  Qnn_Tensor_t outputs__layer1_attn_Squeeze_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Squeeze_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__layer1_attn_Squeeze_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_attn_Squeeze_1", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer1_attn_Squeeze_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer1_attn_Squeeze_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__layer1_attn_Constant_1_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__layer1_attn_Constant_1_output_0[] = {1};
  VALIDATE(model.addTensor("_layer1_attn_Constant_1_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_layer1_attn_Constant_1_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_INT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions__layer1_attn_Constant_1_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_layer1_attn_Constant_1_output_0),
                                                .dataSize=BINLEN(_layer1_attn_Constant_1_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__layer1_attn_Gather_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_attn_Gather_3 */
  Qnn_Param_t params__layer1_attn_Gather_3[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_INT_32, {.int32Value = 0}}}}
  };
  const char*  inputs__layer1_attn_Gather_3[] = {
    "_layer1_attn_Squeeze_1_output_0",
    "_layer1_attn_Constant_1_output_0"
  };
  uint32_t dimensions__layer1_attn_Gather_3_output_0_pre_reshape[] = {1, 8, 1, 64};
  Qnn_Tensor_t outputs__layer1_attn_Gather_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Gather_3_output_0_pre_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__layer1_attn_Gather_3_output_0_pre_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_attn_Gather_3", // Node Name
                         "qti.aisw", // Package Name
                         "Gather", // Qnn Node Type
                         params__layer1_attn_Gather_3, // Node Params
                         1, // Num Node Params
                         inputs__layer1_attn_Gather_3, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__layer1_attn_Gather_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_Reshape_post__layer1_attn_Gather_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR Reshape_post__layer1_attn_Gather_3 */
  const char*  inputs_Reshape_post__layer1_attn_Gather_3[] = {
    "_layer1_attn_Gather_3_output_0_pre_reshape"
  };
  uint32_t dimensions__layer1_attn_Reshape_3_output_0[] = {8, 4, 16};
  Qnn_Tensor_t outputs_Reshape_post__layer1_attn_Gather_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Reshape_3_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer1_attn_Reshape_3_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "Reshape_post__layer1_attn_Gather_3", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs_Reshape_post__layer1_attn_Gather_3, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_Reshape_post__layer1_attn_Gather_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__layer1_attn_Constant_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__layer1_attn_Constant_output_0[] = {1};
  VALIDATE(model.addTensor("_layer1_attn_Constant_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_layer1_attn_Constant_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_INT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions__layer1_attn_Constant_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_layer1_attn_Constant_output_0),
                                                .dataSize=BINLEN(_layer1_attn_Constant_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__layer1_attn_Gather_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_attn_Gather_4 */
  Qnn_Param_t params__layer1_attn_Gather_4[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_INT_32, {.int32Value = 0}}}}
  };
  const char*  inputs__layer1_attn_Gather_4[] = {
    "_layer1_attn_Squeeze_1_output_0",
    "_layer1_attn_Constant_output_0"
  };
  uint32_t dimensions__layer1_attn_Gather_4_output_0_pre_reshape[] = {1, 8, 1, 64};
  Qnn_Tensor_t outputs__layer1_attn_Gather_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Gather_4_output_0_pre_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__layer1_attn_Gather_4_output_0_pre_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_attn_Gather_4", // Node Name
                         "qti.aisw", // Package Name
                         "Gather", // Qnn Node Type
                         params__layer1_attn_Gather_4, // Node Params
                         1, // Num Node Params
                         inputs__layer1_attn_Gather_4, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__layer1_attn_Gather_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_Reshape_post__layer1_attn_Gather_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR Reshape_post__layer1_attn_Gather_4 */
  const char*  inputs_Reshape_post__layer1_attn_Gather_4[] = {
    "_layer1_attn_Gather_4_output_0_pre_reshape"
  };
  uint32_t dimensions__layer1_attn_Reshape_4_output_0[] = {8, 4, 16};
  Qnn_Tensor_t outputs_Reshape_post__layer1_attn_Gather_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Reshape_4_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer1_attn_Reshape_4_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "Reshape_post__layer1_attn_Gather_4", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs_Reshape_post__layer1_attn_Gather_4, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_Reshape_post__layer1_attn_Gather_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__layer1_attn_Constant_4_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__layer1_attn_Constant_4_output_0[] = {1};
  VALIDATE(model.addTensor("_layer1_attn_Constant_4_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_layer1_attn_Constant_4_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_INT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions__layer1_attn_Constant_4_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_layer1_attn_Constant_4_output_0),
                                                .dataSize=BINLEN(_layer1_attn_Constant_4_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__layer1_attn_Gather_5(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_attn_Gather_5 */
  Qnn_Param_t params__layer1_attn_Gather_5[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_INT_32, {.int32Value = 0}}}}
  };
  const char*  inputs__layer1_attn_Gather_5[] = {
    "_layer1_attn_Squeeze_1_output_0",
    "_layer1_attn_Constant_4_output_0"
  };
  uint32_t dimensions__layer1_attn_Gather_5_output_0_pre_reshape[] = {1, 8, 1, 64};
  Qnn_Tensor_t outputs__layer1_attn_Gather_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Gather_5_output_0_pre_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__layer1_attn_Gather_5_output_0_pre_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_attn_Gather_5", // Node Name
                         "qti.aisw", // Package Name
                         "Gather", // Qnn Node Type
                         params__layer1_attn_Gather_5, // Node Params
                         1, // Num Node Params
                         inputs__layer1_attn_Gather_5, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__layer1_attn_Gather_5, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_Reshape_post__layer1_attn_Gather_5(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR Reshape_post__layer1_attn_Gather_5 */
  const char*  inputs_Reshape_post__layer1_attn_Gather_5[] = {
    "_layer1_attn_Gather_5_output_0_pre_reshape"
  };
  uint32_t dimensions__layer1_attn_Reshape_5_output_0[] = {8, 4, 16};
  Qnn_Tensor_t outputs_Reshape_post__layer1_attn_Gather_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Reshape_5_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer1_attn_Reshape_5_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "Reshape_post__layer1_attn_Gather_5", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs_Reshape_post__layer1_attn_Gather_5, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_Reshape_post__layer1_attn_Gather_5, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer1_attn_Transpose_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_attn_Transpose_2 */
  uint32_t dimensions__layer1_attn_Transpose_2_perm[] = {3};
  uint32_t _layer1_attn_Transpose_2_perm[] = {1, 0, 2};
  Qnn_Param_t params__layer1_attn_Transpose_2[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Transpose_2_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__layer1_attn_Transpose_2_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_layer1_attn_Transpose_2_perm,
                           .dataSize=12}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__layer1_attn_Transpose_2[] = {
    "_layer1_attn_Reshape_3_output_0"
  };
  uint32_t dimensions__layer1_attn_Transpose_2_output_0[] = {4, 8, 16};
  Qnn_Tensor_t outputs__layer1_attn_Transpose_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Transpose_2_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer1_attn_Transpose_2_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_attn_Transpose_2", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__layer1_attn_Transpose_2, // Node Params
                         1, // Num Node Params
                         inputs__layer1_attn_Transpose_2, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer1_attn_Transpose_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer1_attn_Transpose_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_attn_Transpose_3 */
  uint32_t dimensions__layer1_attn_Transpose_3_perm[] = {3};
  uint32_t _layer1_attn_Transpose_3_perm[] = {1, 0, 2};
  Qnn_Param_t params__layer1_attn_Transpose_3[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Transpose_3_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__layer1_attn_Transpose_3_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_layer1_attn_Transpose_3_perm,
                           .dataSize=12}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__layer1_attn_Transpose_3[] = {
    "_layer1_attn_Reshape_5_output_0"
  };
  uint32_t dimensions__layer1_attn_Transpose_3_output_0[] = {4, 8, 16};
  Qnn_Tensor_t outputs__layer1_attn_Transpose_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Transpose_3_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer1_attn_Transpose_3_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_attn_Transpose_3", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__layer1_attn_Transpose_3, // Node Params
                         1, // Num Node Params
                         inputs__layer1_attn_Transpose_3, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer1_attn_Transpose_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__layer1_attn_Constant_24_output_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__layer1_attn_Constant_24_output_0[] = {1};
  VALIDATE(model.addTensor("_layer1_attn_Constant_24_output_0", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_layer1_attn_Constant_24_output_0",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions__layer1_attn_Constant_24_output_0,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_layer1_attn_Constant_24_output_0),
                                                .dataSize=BINLEN(_layer1_attn_Constant_24_output_0)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__layer1_attn_Mul_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_attn_Mul_1 */
  Qnn_Param_t params__layer1_attn_Mul_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs__layer1_attn_Mul_1[] = {
    "_layer1_attn_Transpose_2_output_0",
    "_layer1_attn_Constant_24_output_0"
  };
  uint32_t dimensions__layer1_attn_Mul_1_output_0[] = {4, 8, 16};
  Qnn_Tensor_t outputs__layer1_attn_Mul_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Mul_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer1_attn_Mul_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_attn_Mul_1", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__layer1_attn_Mul_1, // Node Params
                         1, // Num Node Params
                         inputs__layer1_attn_Mul_1, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__layer1_attn_Mul_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer1_attn_Transpose_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_attn_Transpose_4 */
  uint32_t dimensions__layer1_attn_Transpose_4_perm[] = {3};
  uint32_t _layer1_attn_Transpose_4_perm[] = {1, 2, 0};
  Qnn_Param_t params__layer1_attn_Transpose_4[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Transpose_4_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__layer1_attn_Transpose_4_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_layer1_attn_Transpose_4_perm,
                           .dataSize=12}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__layer1_attn_Transpose_4[] = {
    "_layer1_attn_Reshape_4_output_0"
  };
  uint32_t dimensions__layer1_attn_Transpose_4_output_0[] = {4, 16, 8};
  Qnn_Tensor_t outputs__layer1_attn_Transpose_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Transpose_4_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer1_attn_Transpose_4_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_attn_Transpose_4", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__layer1_attn_Transpose_4, // Node Params
                         1, // Num Node Params
                         inputs__layer1_attn_Transpose_4, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer1_attn_Transpose_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer1_attn_MatMul_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_attn_MatMul_1 */
  Qnn_Param_t params__layer1_attn_MatMul_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs__layer1_attn_MatMul_1[] = {
    "_layer1_attn_Mul_1_output_0",
    "_layer1_attn_Transpose_4_output_0"
  };
  uint32_t dimensions__layer1_attn_MatMul_1_output_0[] = {4, 8, 8};
  Qnn_Tensor_t outputs__layer1_attn_MatMul_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_MatMul_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer1_attn_MatMul_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_attn_MatMul_1", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params__layer1_attn_MatMul_1, // Node Params
                         2, // Num Node Params
                         inputs__layer1_attn_MatMul_1, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__layer1_attn_MatMul_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer1_attn_Softmax(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_attn_Softmax */
  Qnn_Param_t params__layer1_attn_Softmax[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 2}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="beta",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 1.0000000000000000000000000000000000000000f}}}}
  };
  const char*  inputs__layer1_attn_Softmax[] = {
    "_layer1_attn_MatMul_1_output_0"
  };
  uint32_t dimensions__layer1_attn_Softmax_output_0[] = {4, 8, 8};
  Qnn_Tensor_t outputs__layer1_attn_Softmax[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Softmax_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer1_attn_Softmax_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_attn_Softmax", // Node Name
                         "qti.aisw", // Package Name
                         "Softmax", // Qnn Node Type
                         params__layer1_attn_Softmax, // Node Params
                         2, // Num Node Params
                         inputs__layer1_attn_Softmax, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer1_attn_Softmax, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer1_attn_MatMul_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_attn_MatMul_2 */
  Qnn_Param_t params__layer1_attn_MatMul_2[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs__layer1_attn_MatMul_2[] = {
    "_layer1_attn_Softmax_output_0",
    "_layer1_attn_Transpose_3_output_0"
  };
  uint32_t dimensions__layer1_attn_MatMul_2_output_0[] = {4, 8, 16};
  Qnn_Tensor_t outputs__layer1_attn_MatMul_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_MatMul_2_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer1_attn_MatMul_2_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_attn_MatMul_2", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params__layer1_attn_MatMul_2, // Node Params
                         2, // Num Node Params
                         inputs__layer1_attn_MatMul_2, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__layer1_attn_MatMul_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer1_attn_Transpose_5(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_attn_Transpose_5 */
  uint32_t dimensions__layer1_attn_Transpose_5_perm[] = {3};
  uint32_t _layer1_attn_Transpose_5_perm[] = {1, 0, 2};
  Qnn_Param_t params__layer1_attn_Transpose_5[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Transpose_5_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__layer1_attn_Transpose_5_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_layer1_attn_Transpose_5_perm,
                           .dataSize=12}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__layer1_attn_Transpose_5[] = {
    "_layer1_attn_MatMul_2_output_0"
  };
  uint32_t dimensions__layer1_attn_Transpose_5_output_0[] = {8, 4, 16};
  Qnn_Tensor_t outputs__layer1_attn_Transpose_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Transpose_5_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer1_attn_Transpose_5_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_attn_Transpose_5", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__layer1_attn_Transpose_5, // Node Params
                         1, // Num Node Params
                         inputs__layer1_attn_Transpose_5, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer1_attn_Transpose_5, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_layer1_attn_out_proj_weight_permute(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_layer1_attn_out_proj_weight_permute[] = {64, 64};
  VALIDATE(model.addTensor("layer1_attn_out_proj_weight_permute", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "layer1_attn_out_proj_weight_permute",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_layer1_attn_out_proj_weight_permute,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(layer1_attn_out_proj_weight_permute),
                                                .dataSize=BINLEN(layer1_attn_out_proj_weight_permute)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_layer1_attn_out_proj_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_layer1_attn_out_proj_bias[] = {64};
  VALIDATE(model.addTensor("layer1_attn_out_proj_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "layer1_attn_out_proj_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_layer1_attn_out_proj_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(layer1_attn_out_proj_bias),
                                                .dataSize=BINLEN(layer1_attn_out_proj_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__layer1_attn_Gemm(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_attn_Gemm */
  const char*  inputs__layer1_attn_Gemm[] = {
    "_layer1_attn_Transpose_5_output_0",
    "layer1_attn_out_proj_weight_permute",
    "layer1_attn_out_proj_bias"
  };
  uint32_t dimensions__layer1_attn_Gemm_output_0[] = {8, 64};
  Qnn_Tensor_t outputs__layer1_attn_Gemm[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Gemm_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__layer1_attn_Gemm_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_attn_Gemm", // Node Name
                         "qti.aisw", // Package Name
                         "FullyConnected", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer1_attn_Gemm, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__layer1_attn_Gemm, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer1_attn_Reshape_7(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_attn_Reshape_7 */
  const char*  inputs__layer1_attn_Reshape_7[] = {
    "_layer1_attn_Gemm_output_0"
  };
  uint32_t dimensions__layer1_attn_Reshape_7_output_0[] = {8, 1, 64};
  Qnn_Tensor_t outputs__layer1_attn_Reshape_7[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Reshape_7_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer1_attn_Reshape_7_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_attn_Reshape_7", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer1_attn_Reshape_7, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer1_attn_Reshape_7, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer1_attn_Transpose_6(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_attn_Transpose_6 */
  uint32_t dimensions__layer1_attn_Transpose_6_perm[] = {3};
  uint32_t _layer1_attn_Transpose_6_perm[] = {1, 0, 2};
  Qnn_Param_t params__layer1_attn_Transpose_6[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Transpose_6_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__layer1_attn_Transpose_6_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_layer1_attn_Transpose_6_perm,
                           .dataSize=12}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__layer1_attn_Transpose_6[] = {
    "_layer1_attn_Reshape_7_output_0"
  };
  uint32_t dimensions__layer1_attn_Transpose_6_output_0[] = {1, 8, 64};
  Qnn_Tensor_t outputs__layer1_attn_Transpose_6[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_attn_Transpose_6_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer1_attn_Transpose_6_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_attn_Transpose_6", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__layer1_attn_Transpose_6, // Node Params
                         1, // Num Node Params
                         inputs__layer1_attn_Transpose_6, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer1_attn_Transpose_6, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer1_Add(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_Add */
  Qnn_Param_t params__layer1_Add[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__layer1_Add[] = {
    "input_ncf",
    "_layer1_attn_Transpose_6_output_0"
  };
  uint32_t dimensions__layer1_Add_output_0[] = {1, 8, 64};
  Qnn_Tensor_t outputs__layer1_Add[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_Add_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer1_Add_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_Add", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__layer1_Add, // Node Params
                         1, // Num Node Params
                         inputs__layer1_Add, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__layer1_Add, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_layer1_norm1_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_layer1_norm1_weight[] = {64};
  VALIDATE(model.addTensor("layer1_norm1_weight", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "layer1_norm1_weight",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_layer1_norm1_weight,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(layer1_norm1_weight),
                                                .dataSize=BINLEN(layer1_norm1_weight)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__layer1_norm1_LayerNormalization(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_norm1_LayerNormalization */
  uint32_t dimensions__layer1_norm1_LayerNormalization_axes[] = {1};
  uint32_t _layer1_norm1_LayerNormalization_axes[] = {2};
  Qnn_Param_t params__layer1_norm1_LayerNormalization[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="axes",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_norm1_LayerNormalization_axes",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__layer1_norm1_LayerNormalization_axes,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_layer1_norm1_LayerNormalization_axes,
                           .dataSize=4}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="epsilon",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.0000099999997473787516355514526367187500f}}}}
  };
  const char*  inputs__layer1_norm1_LayerNormalization[] = {
    "_layer1_Add_output_0",
    "layer1_norm1_weight",
    "layer1_attn_out_proj_bias"
  };
  uint32_t dimensions__layer1_norm1_LayerNormalization_output_0[] = {1, 8, 64};
  Qnn_Tensor_t outputs__layer1_norm1_LayerNormalization[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_norm1_LayerNormalization_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer1_norm1_LayerNormalization_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_norm1_LayerNormalization", // Node Name
                         "qti.aisw", // Package Name
                         "LayerNorm", // Qnn Node Type
                         params__layer1_norm1_LayerNormalization, // Node Params
                         2, // Num Node Params
                         inputs__layer1_norm1_LayerNormalization, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__layer1_norm1_LayerNormalization, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer1_ff_ff_0_MatMul_pre_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_ff_ff_0_MatMul_pre_reshape */
  const char*  inputs__layer1_ff_ff_0_MatMul_pre_reshape[] = {
    "_layer1_norm1_LayerNormalization_output_0"
  };
  uint32_t dimensions__layer1_ff_ff_0_MatMul_pre_reshape[] = {8, 64};
  Qnn_Tensor_t outputs__layer1_ff_ff_0_MatMul_pre_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_ff_ff_0_MatMul_pre_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__layer1_ff_ff_0_MatMul_pre_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_ff_ff_0_MatMul_pre_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer1_ff_ff_0_MatMul_pre_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer1_ff_ff_0_MatMul_pre_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__MatMul_294(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__MatMul_294[] = {128, 64};
  VALIDATE(model.addTensor("onnx__MatMul_294", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__MatMul_294",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_onnx__MatMul_294,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__MatMul_294),
                                                .dataSize=BINLEN(onnx__MatMul_294)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_layer1_ff_0_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_layer1_ff_0_bias[] = {128};
  VALIDATE(model.addTensor("layer1_ff_0_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "layer1_ff_0_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_layer1_ff_0_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(layer1_ff_0_bias),
                                                .dataSize=BINLEN(layer1_ff_0_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__layer1_ff_ff_0_MatMul(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_ff_ff_0_MatMul */
  const char*  inputs__layer1_ff_ff_0_MatMul[] = {
    "_layer1_ff_ff_0_MatMul_pre_reshape",
    "onnx__MatMul_294",
    "layer1_ff_0_bias"
  };
  uint32_t dimensions__layer1_ff_ff_0_Add_output_0_fc[] = {8, 128};
  Qnn_Tensor_t outputs__layer1_ff_ff_0_MatMul[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_ff_ff_0_Add_output_0_fc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__layer1_ff_ff_0_Add_output_0_fc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_ff_ff_0_MatMul", // Node Name
                         "qti.aisw", // Package Name
                         "FullyConnected", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer1_ff_ff_0_MatMul, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__layer1_ff_ff_0_MatMul, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer1_ff_ff_0_MatMul_post_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_ff_ff_0_MatMul_post_reshape */
  const char*  inputs__layer1_ff_ff_0_MatMul_post_reshape[] = {
    "_layer1_ff_ff_0_Add_output_0_fc"
  };
  uint32_t dimensions__layer1_ff_ff_0_Add_output_0[] = {1, 8, 128};
  Qnn_Tensor_t outputs__layer1_ff_ff_0_MatMul_post_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_ff_ff_0_Add_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer1_ff_ff_0_Add_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_ff_ff_0_MatMul_post_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer1_ff_ff_0_MatMul_post_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer1_ff_ff_0_MatMul_post_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__elementwiseneuron_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _elementwiseneuron_1 */
  Qnn_Param_t params__elementwiseneuron_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__elementwiseneuron_1[] = {
    "_layer1_ff_ff_0_Add_output_0"
  };
  uint32_t dimensions__layer1_ff_ff_1_Mul_1_output_0[] = {1, 8, 128};
  Qnn_Tensor_t outputs__elementwiseneuron_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_ff_ff_1_Mul_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer1_ff_ff_1_Mul_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_elementwiseneuron_1", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__elementwiseneuron_1, // Node Params
                         1, // Num Node Params
                         inputs__elementwiseneuron_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__elementwiseneuron_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer1_ff_ff_2_MatMul_pre_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_ff_ff_2_MatMul_pre_reshape */
  const char*  inputs__layer1_ff_ff_2_MatMul_pre_reshape[] = {
    "_layer1_ff_ff_1_Mul_1_output_0"
  };
  uint32_t dimensions__layer1_ff_ff_2_MatMul_pre_reshape[] = {8, 128};
  Qnn_Tensor_t outputs__layer1_ff_ff_2_MatMul_pre_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_ff_ff_2_MatMul_pre_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__layer1_ff_ff_2_MatMul_pre_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_ff_ff_2_MatMul_pre_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer1_ff_ff_2_MatMul_pre_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer1_ff_ff_2_MatMul_pre_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__MatMul_295(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__MatMul_295[] = {64, 128};
  VALIDATE(model.addTensor("onnx__MatMul_295", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__MatMul_295",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_onnx__MatMul_295,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__MatMul_295),
                                                .dataSize=BINLEN(onnx__MatMul_295)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_layer1_ff_2_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_layer1_ff_2_bias[] = {64};
  VALIDATE(model.addTensor("layer1_ff_2_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "layer1_ff_2_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_layer1_ff_2_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(layer1_ff_2_bias),
                                                .dataSize=BINLEN(layer1_ff_2_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__layer1_ff_ff_2_MatMul(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_ff_ff_2_MatMul */
  const char*  inputs__layer1_ff_ff_2_MatMul[] = {
    "_layer1_ff_ff_2_MatMul_pre_reshape",
    "onnx__MatMul_295",
    "layer1_ff_2_bias"
  };
  uint32_t dimensions__layer1_ff_ff_2_Add_output_0_fc[] = {8, 64};
  Qnn_Tensor_t outputs__layer1_ff_ff_2_MatMul[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_ff_ff_2_Add_output_0_fc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__layer1_ff_ff_2_Add_output_0_fc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_ff_ff_2_MatMul", // Node Name
                         "qti.aisw", // Package Name
                         "FullyConnected", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer1_ff_ff_2_MatMul, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__layer1_ff_ff_2_MatMul, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer1_ff_ff_2_MatMul_post_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_ff_ff_2_MatMul_post_reshape */
  const char*  inputs__layer1_ff_ff_2_MatMul_post_reshape[] = {
    "_layer1_ff_ff_2_Add_output_0_fc"
  };
  uint32_t dimensions__layer1_ff_ff_2_Add_output_0[] = {1, 8, 64};
  Qnn_Tensor_t outputs__layer1_ff_ff_2_MatMul_post_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_ff_ff_2_Add_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer1_ff_ff_2_Add_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_ff_ff_2_MatMul_post_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer1_ff_ff_2_MatMul_post_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer1_ff_ff_2_MatMul_post_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer1_Add_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_Add_1 */
  Qnn_Param_t params__layer1_Add_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__layer1_Add_1[] = {
    "_layer1_norm1_LayerNormalization_output_0",
    "_layer1_ff_ff_2_Add_output_0"
  };
  uint32_t dimensions__layer1_Add_1_output_0[] = {1, 8, 64};
  Qnn_Tensor_t outputs__layer1_Add_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_Add_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer1_Add_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_Add_1", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__layer1_Add_1, // Node Params
                         1, // Num Node Params
                         inputs__layer1_Add_1, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__layer1_Add_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer1_norm2_LayerNormalization(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer1_norm2_LayerNormalization */
  uint32_t dimensions__layer1_norm2_LayerNormalization_axes[] = {1};
  uint32_t _layer1_norm2_LayerNormalization_axes[] = {2};
  Qnn_Param_t params__layer1_norm2_LayerNormalization[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="axes",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_norm2_LayerNormalization_axes",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__layer1_norm2_LayerNormalization_axes,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_layer1_norm2_LayerNormalization_axes,
                           .dataSize=4}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="epsilon",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.0000099999997473787516355514526367187500f}}}}
  };
  const char*  inputs__layer1_norm2_LayerNormalization[] = {
    "_layer1_Add_1_output_0",
    "layer1_norm1_weight",
    "layer1_attn_out_proj_bias"
  };
  uint32_t dimensions__layer1_norm2_LayerNormalization_output_0[] = {1, 8, 64};
  Qnn_Tensor_t outputs__layer1_norm2_LayerNormalization[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer1_norm2_LayerNormalization_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer1_norm2_LayerNormalization_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer1_norm2_LayerNormalization", // Node Name
                         "qti.aisw", // Package Name
                         "LayerNorm", // Qnn Node Type
                         params__layer1_norm2_LayerNormalization, // Node Params
                         2, // Num Node Params
                         inputs__layer1_norm2_LayerNormalization, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__layer1_norm2_LayerNormalization, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_attn_Transpose(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_attn_Transpose */
  uint32_t dimensions__layer2_attn_Transpose_perm[] = {3};
  uint32_t _layer2_attn_Transpose_perm[] = {1, 0, 2};
  Qnn_Param_t params__layer2_attn_Transpose[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Transpose_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__layer2_attn_Transpose_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_layer2_attn_Transpose_perm,
                           .dataSize=12}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__layer2_attn_Transpose[] = {
    "_layer1_norm2_LayerNormalization_output_0"
  };
  uint32_t dimensions__layer2_attn_Transpose_output_0[] = {8, 1, 64};
  Qnn_Tensor_t outputs__layer2_attn_Transpose[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Transpose_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer2_attn_Transpose_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_attn_Transpose", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__layer2_attn_Transpose, // Node Params
                         1, // Num Node Params
                         inputs__layer2_attn_Transpose, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer2_attn_Transpose, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__MatMul_296(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__MatMul_296[] = {192, 64};
  VALIDATE(model.addTensor("onnx__MatMul_296", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__MatMul_296",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_onnx__MatMul_296,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__MatMul_296),
                                                .dataSize=BINLEN(onnx__MatMul_296)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__layer2_attn_MatMul(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_attn_MatMul */
  const char*  inputs__layer2_attn_MatMul[] = {
    "_layer2_attn_Transpose_output_0",
    "onnx__MatMul_296"
  };
  uint32_t dimensions__layer2_attn_MatMul_output_0_fc[] = {8, 192};
  Qnn_Tensor_t outputs__layer2_attn_MatMul[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_MatMul_output_0_fc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__layer2_attn_MatMul_output_0_fc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_attn_MatMul", // Node Name
                         "qti.aisw", // Package Name
                         "FullyConnected", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer2_attn_MatMul, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__layer2_attn_MatMul, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_attn_MatMul_post_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_attn_MatMul_post_reshape */
  const char*  inputs__layer2_attn_MatMul_post_reshape[] = {
    "_layer2_attn_MatMul_output_0_fc"
  };
  uint32_t dimensions__layer2_attn_Unsqueeze_1_output_0[] = {1, 8, 1, 3, 64};
  Qnn_Tensor_t outputs__layer2_attn_MatMul_post_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Unsqueeze_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__layer2_attn_Unsqueeze_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_attn_MatMul_post_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer2_attn_MatMul_post_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer2_attn_MatMul_post_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_attn_Transpose_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_attn_Transpose_1 */
  uint32_t dimensions__layer2_attn_Transpose_1_perm[] = {5};
  uint32_t _layer2_attn_Transpose_1_perm[] = {3, 1, 2, 0, 4};
  Qnn_Param_t params__layer2_attn_Transpose_1[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Transpose_1_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__layer2_attn_Transpose_1_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_layer2_attn_Transpose_1_perm,
                           .dataSize=20}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__layer2_attn_Transpose_1[] = {
    "_layer2_attn_Unsqueeze_1_output_0"
  };
  uint32_t dimensions__layer2_attn_Transpose_1_output_0[] = {3, 8, 1, 1, 64};
  Qnn_Tensor_t outputs__layer2_attn_Transpose_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Transpose_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 5,
            .dimensions=dimensions__layer2_attn_Transpose_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_attn_Transpose_1", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__layer2_attn_Transpose_1, // Node Params
                         1, // Num Node Params
                         inputs__layer2_attn_Transpose_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer2_attn_Transpose_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_attn_Squeeze_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_attn_Squeeze_1 */
  const char*  inputs__layer2_attn_Squeeze_1[] = {
    "_layer2_attn_Transpose_1_output_0"
  };
  uint32_t dimensions__layer2_attn_Squeeze_1_output_0[] = {3, 8, 1, 64};
  Qnn_Tensor_t outputs__layer2_attn_Squeeze_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Squeeze_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__layer2_attn_Squeeze_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_attn_Squeeze_1", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer2_attn_Squeeze_1, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer2_attn_Squeeze_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_attn_Gather_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_attn_Gather_3 */
  Qnn_Param_t params__layer2_attn_Gather_3[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_INT_32, {.int32Value = 0}}}}
  };
  const char*  inputs__layer2_attn_Gather_3[] = {
    "_layer2_attn_Squeeze_1_output_0",
    "_layer1_attn_Constant_1_output_0"
  };
  uint32_t dimensions__layer2_attn_Gather_3_output_0_pre_reshape[] = {1, 8, 1, 64};
  Qnn_Tensor_t outputs__layer2_attn_Gather_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Gather_3_output_0_pre_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__layer2_attn_Gather_3_output_0_pre_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_attn_Gather_3", // Node Name
                         "qti.aisw", // Package Name
                         "Gather", // Qnn Node Type
                         params__layer2_attn_Gather_3, // Node Params
                         1, // Num Node Params
                         inputs__layer2_attn_Gather_3, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__layer2_attn_Gather_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_Reshape_post__layer2_attn_Gather_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR Reshape_post__layer2_attn_Gather_3 */
  const char*  inputs_Reshape_post__layer2_attn_Gather_3[] = {
    "_layer2_attn_Gather_3_output_0_pre_reshape"
  };
  uint32_t dimensions__layer2_attn_Reshape_3_output_0[] = {8, 4, 16};
  Qnn_Tensor_t outputs_Reshape_post__layer2_attn_Gather_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Reshape_3_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer2_attn_Reshape_3_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "Reshape_post__layer2_attn_Gather_3", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs_Reshape_post__layer2_attn_Gather_3, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_Reshape_post__layer2_attn_Gather_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_attn_Gather_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_attn_Gather_4 */
  Qnn_Param_t params__layer2_attn_Gather_4[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_INT_32, {.int32Value = 0}}}}
  };
  const char*  inputs__layer2_attn_Gather_4[] = {
    "_layer2_attn_Squeeze_1_output_0",
    "_layer1_attn_Constant_output_0"
  };
  uint32_t dimensions__layer2_attn_Gather_4_output_0_pre_reshape[] = {1, 8, 1, 64};
  Qnn_Tensor_t outputs__layer2_attn_Gather_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Gather_4_output_0_pre_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__layer2_attn_Gather_4_output_0_pre_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_attn_Gather_4", // Node Name
                         "qti.aisw", // Package Name
                         "Gather", // Qnn Node Type
                         params__layer2_attn_Gather_4, // Node Params
                         1, // Num Node Params
                         inputs__layer2_attn_Gather_4, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__layer2_attn_Gather_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_Reshape_post__layer2_attn_Gather_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR Reshape_post__layer2_attn_Gather_4 */
  const char*  inputs_Reshape_post__layer2_attn_Gather_4[] = {
    "_layer2_attn_Gather_4_output_0_pre_reshape"
  };
  uint32_t dimensions__layer2_attn_Reshape_4_output_0[] = {8, 4, 16};
  Qnn_Tensor_t outputs_Reshape_post__layer2_attn_Gather_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Reshape_4_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer2_attn_Reshape_4_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "Reshape_post__layer2_attn_Gather_4", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs_Reshape_post__layer2_attn_Gather_4, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_Reshape_post__layer2_attn_Gather_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_attn_Gather_5(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_attn_Gather_5 */
  Qnn_Param_t params__layer2_attn_Gather_5[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_INT_32, {.int32Value = 0}}}}
  };
  const char*  inputs__layer2_attn_Gather_5[] = {
    "_layer2_attn_Squeeze_1_output_0",
    "_layer1_attn_Constant_4_output_0"
  };
  uint32_t dimensions__layer2_attn_Gather_5_output_0_pre_reshape[] = {1, 8, 1, 64};
  Qnn_Tensor_t outputs__layer2_attn_Gather_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Gather_5_output_0_pre_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__layer2_attn_Gather_5_output_0_pre_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_attn_Gather_5", // Node Name
                         "qti.aisw", // Package Name
                         "Gather", // Qnn Node Type
                         params__layer2_attn_Gather_5, // Node Params
                         1, // Num Node Params
                         inputs__layer2_attn_Gather_5, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__layer2_attn_Gather_5, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_Reshape_post__layer2_attn_Gather_5(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR Reshape_post__layer2_attn_Gather_5 */
  const char*  inputs_Reshape_post__layer2_attn_Gather_5[] = {
    "_layer2_attn_Gather_5_output_0_pre_reshape"
  };
  uint32_t dimensions__layer2_attn_Reshape_5_output_0[] = {8, 4, 16};
  Qnn_Tensor_t outputs_Reshape_post__layer2_attn_Gather_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Reshape_5_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer2_attn_Reshape_5_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "Reshape_post__layer2_attn_Gather_5", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs_Reshape_post__layer2_attn_Gather_5, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_Reshape_post__layer2_attn_Gather_5, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_attn_Transpose_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_attn_Transpose_2 */
  uint32_t dimensions__layer2_attn_Transpose_2_perm[] = {3};
  uint32_t _layer2_attn_Transpose_2_perm[] = {1, 0, 2};
  Qnn_Param_t params__layer2_attn_Transpose_2[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Transpose_2_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__layer2_attn_Transpose_2_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_layer2_attn_Transpose_2_perm,
                           .dataSize=12}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__layer2_attn_Transpose_2[] = {
    "_layer2_attn_Reshape_3_output_0"
  };
  uint32_t dimensions__layer2_attn_Transpose_2_output_0[] = {4, 8, 16};
  Qnn_Tensor_t outputs__layer2_attn_Transpose_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Transpose_2_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer2_attn_Transpose_2_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_attn_Transpose_2", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__layer2_attn_Transpose_2, // Node Params
                         1, // Num Node Params
                         inputs__layer2_attn_Transpose_2, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer2_attn_Transpose_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_attn_Transpose_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_attn_Transpose_3 */
  uint32_t dimensions__layer2_attn_Transpose_3_perm[] = {3};
  uint32_t _layer2_attn_Transpose_3_perm[] = {1, 0, 2};
  Qnn_Param_t params__layer2_attn_Transpose_3[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Transpose_3_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__layer2_attn_Transpose_3_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_layer2_attn_Transpose_3_perm,
                           .dataSize=12}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__layer2_attn_Transpose_3[] = {
    "_layer2_attn_Reshape_5_output_0"
  };
  uint32_t dimensions__layer2_attn_Transpose_3_output_0[] = {4, 8, 16};
  Qnn_Tensor_t outputs__layer2_attn_Transpose_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Transpose_3_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer2_attn_Transpose_3_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_attn_Transpose_3", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__layer2_attn_Transpose_3, // Node Params
                         1, // Num Node Params
                         inputs__layer2_attn_Transpose_3, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer2_attn_Transpose_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_attn_Mul_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_attn_Mul_1 */
  Qnn_Param_t params__layer2_attn_Mul_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 13}}}}
  };
  const char*  inputs__layer2_attn_Mul_1[] = {
    "_layer2_attn_Transpose_2_output_0",
    "_layer1_attn_Constant_24_output_0"
  };
  uint32_t dimensions__layer2_attn_Mul_1_output_0[] = {4, 8, 16};
  Qnn_Tensor_t outputs__layer2_attn_Mul_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Mul_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer2_attn_Mul_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_attn_Mul_1", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__layer2_attn_Mul_1, // Node Params
                         1, // Num Node Params
                         inputs__layer2_attn_Mul_1, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__layer2_attn_Mul_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_attn_Transpose_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_attn_Transpose_4 */
  uint32_t dimensions__layer2_attn_Transpose_4_perm[] = {3};
  uint32_t _layer2_attn_Transpose_4_perm[] = {1, 2, 0};
  Qnn_Param_t params__layer2_attn_Transpose_4[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Transpose_4_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__layer2_attn_Transpose_4_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_layer2_attn_Transpose_4_perm,
                           .dataSize=12}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__layer2_attn_Transpose_4[] = {
    "_layer2_attn_Reshape_4_output_0"
  };
  uint32_t dimensions__layer2_attn_Transpose_4_output_0[] = {4, 16, 8};
  Qnn_Tensor_t outputs__layer2_attn_Transpose_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Transpose_4_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer2_attn_Transpose_4_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_attn_Transpose_4", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__layer2_attn_Transpose_4, // Node Params
                         1, // Num Node Params
                         inputs__layer2_attn_Transpose_4, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer2_attn_Transpose_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_attn_MatMul_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_attn_MatMul_1 */
  Qnn_Param_t params__layer2_attn_MatMul_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs__layer2_attn_MatMul_1[] = {
    "_layer2_attn_Mul_1_output_0",
    "_layer2_attn_Transpose_4_output_0"
  };
  uint32_t dimensions__layer2_attn_MatMul_1_output_0[] = {4, 8, 8};
  Qnn_Tensor_t outputs__layer2_attn_MatMul_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_MatMul_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer2_attn_MatMul_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_attn_MatMul_1", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params__layer2_attn_MatMul_1, // Node Params
                         2, // Num Node Params
                         inputs__layer2_attn_MatMul_1, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__layer2_attn_MatMul_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_attn_Softmax(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_attn_Softmax */
  Qnn_Param_t params__layer2_attn_Softmax[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 2}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="beta",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 1.0000000000000000000000000000000000000000f}}}}
  };
  const char*  inputs__layer2_attn_Softmax[] = {
    "_layer2_attn_MatMul_1_output_0"
  };
  uint32_t dimensions__layer2_attn_Softmax_output_0[] = {4, 8, 8};
  Qnn_Tensor_t outputs__layer2_attn_Softmax[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Softmax_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer2_attn_Softmax_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_attn_Softmax", // Node Name
                         "qti.aisw", // Package Name
                         "Softmax", // Qnn Node Type
                         params__layer2_attn_Softmax, // Node Params
                         2, // Num Node Params
                         inputs__layer2_attn_Softmax, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer2_attn_Softmax, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_attn_MatMul_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_attn_MatMul_2 */
  Qnn_Param_t params__layer2_attn_MatMul_2[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in0",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="transpose_in1",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs__layer2_attn_MatMul_2[] = {
    "_layer2_attn_Softmax_output_0",
    "_layer2_attn_Transpose_3_output_0"
  };
  uint32_t dimensions__layer2_attn_MatMul_2_output_0[] = {4, 8, 16};
  Qnn_Tensor_t outputs__layer2_attn_MatMul_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_MatMul_2_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer2_attn_MatMul_2_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_attn_MatMul_2", // Node Name
                         "qti.aisw", // Package Name
                         "MatMul", // Qnn Node Type
                         params__layer2_attn_MatMul_2, // Node Params
                         2, // Num Node Params
                         inputs__layer2_attn_MatMul_2, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__layer2_attn_MatMul_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_attn_Transpose_5(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_attn_Transpose_5 */
  uint32_t dimensions__layer2_attn_Transpose_5_perm[] = {3};
  uint32_t _layer2_attn_Transpose_5_perm[] = {1, 0, 2};
  Qnn_Param_t params__layer2_attn_Transpose_5[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Transpose_5_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__layer2_attn_Transpose_5_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_layer2_attn_Transpose_5_perm,
                           .dataSize=12}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__layer2_attn_Transpose_5[] = {
    "_layer2_attn_MatMul_2_output_0"
  };
  uint32_t dimensions__layer2_attn_Transpose_5_output_0[] = {8, 4, 16};
  Qnn_Tensor_t outputs__layer2_attn_Transpose_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Transpose_5_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer2_attn_Transpose_5_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_attn_Transpose_5", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__layer2_attn_Transpose_5, // Node Params
                         1, // Num Node Params
                         inputs__layer2_attn_Transpose_5, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer2_attn_Transpose_5, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_layer2_attn_out_proj_weight_permute(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_layer2_attn_out_proj_weight_permute[] = {64, 64};
  VALIDATE(model.addTensor("layer2_attn_out_proj_weight_permute", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "layer2_attn_out_proj_weight_permute",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_layer2_attn_out_proj_weight_permute,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(layer2_attn_out_proj_weight_permute),
                                                .dataSize=BINLEN(layer2_attn_out_proj_weight_permute)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__layer2_attn_Gemm(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_attn_Gemm */
  const char*  inputs__layer2_attn_Gemm[] = {
    "_layer2_attn_Transpose_5_output_0",
    "layer2_attn_out_proj_weight_permute",
    "layer1_attn_out_proj_bias"
  };
  uint32_t dimensions__layer2_attn_Gemm_output_0[] = {8, 64};
  Qnn_Tensor_t outputs__layer2_attn_Gemm[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Gemm_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__layer2_attn_Gemm_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_attn_Gemm", // Node Name
                         "qti.aisw", // Package Name
                         "FullyConnected", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer2_attn_Gemm, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__layer2_attn_Gemm, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_attn_Reshape_7(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_attn_Reshape_7 */
  const char*  inputs__layer2_attn_Reshape_7[] = {
    "_layer2_attn_Gemm_output_0"
  };
  uint32_t dimensions__layer2_attn_Reshape_7_output_0[] = {8, 1, 64};
  Qnn_Tensor_t outputs__layer2_attn_Reshape_7[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Reshape_7_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer2_attn_Reshape_7_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_attn_Reshape_7", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer2_attn_Reshape_7, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer2_attn_Reshape_7, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_attn_Transpose_6(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_attn_Transpose_6 */
  uint32_t dimensions__layer2_attn_Transpose_6_perm[] = {3};
  uint32_t _layer2_attn_Transpose_6_perm[] = {1, 0, 2};
  Qnn_Param_t params__layer2_attn_Transpose_6[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Transpose_6_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__layer2_attn_Transpose_6_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_layer2_attn_Transpose_6_perm,
                           .dataSize=12}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs__layer2_attn_Transpose_6[] = {
    "_layer2_attn_Reshape_7_output_0"
  };
  uint32_t dimensions__layer2_attn_Transpose_6_output_0[] = {1, 8, 64};
  Qnn_Tensor_t outputs__layer2_attn_Transpose_6[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_attn_Transpose_6_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer2_attn_Transpose_6_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_attn_Transpose_6", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params__layer2_attn_Transpose_6, // Node Params
                         1, // Num Node Params
                         inputs__layer2_attn_Transpose_6, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer2_attn_Transpose_6, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_Add(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_Add */
  Qnn_Param_t params__layer2_Add[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__layer2_Add[] = {
    "_layer1_norm2_LayerNormalization_output_0",
    "_layer2_attn_Transpose_6_output_0"
  };
  uint32_t dimensions__layer2_Add_output_0[] = {1, 8, 64};
  Qnn_Tensor_t outputs__layer2_Add[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_Add_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer2_Add_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_Add", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__layer2_Add, // Node Params
                         1, // Num Node Params
                         inputs__layer2_Add, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__layer2_Add, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_norm1_LayerNormalization(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_norm1_LayerNormalization */
  uint32_t dimensions__layer2_norm1_LayerNormalization_axes[] = {1};
  uint32_t _layer2_norm1_LayerNormalization_axes[] = {2};
  Qnn_Param_t params__layer2_norm1_LayerNormalization[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="axes",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_norm1_LayerNormalization_axes",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__layer2_norm1_LayerNormalization_axes,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_layer2_norm1_LayerNormalization_axes,
                           .dataSize=4}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="epsilon",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.0000099999997473787516355514526367187500f}}}}
  };
  const char*  inputs__layer2_norm1_LayerNormalization[] = {
    "_layer2_Add_output_0",
    "layer1_norm1_weight",
    "layer1_attn_out_proj_bias"
  };
  uint32_t dimensions__layer2_norm1_LayerNormalization_output_0[] = {1, 8, 64};
  Qnn_Tensor_t outputs__layer2_norm1_LayerNormalization[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_norm1_LayerNormalization_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer2_norm1_LayerNormalization_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_norm1_LayerNormalization", // Node Name
                         "qti.aisw", // Package Name
                         "LayerNorm", // Qnn Node Type
                         params__layer2_norm1_LayerNormalization, // Node Params
                         2, // Num Node Params
                         inputs__layer2_norm1_LayerNormalization, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__layer2_norm1_LayerNormalization, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_ff_ff_0_MatMul_pre_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_ff_ff_0_MatMul_pre_reshape */
  const char*  inputs__layer2_ff_ff_0_MatMul_pre_reshape[] = {
    "_layer2_norm1_LayerNormalization_output_0"
  };
  uint32_t dimensions__layer2_ff_ff_0_MatMul_pre_reshape[] = {8, 64};
  Qnn_Tensor_t outputs__layer2_ff_ff_0_MatMul_pre_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_ff_ff_0_MatMul_pre_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__layer2_ff_ff_0_MatMul_pre_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_ff_ff_0_MatMul_pre_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer2_ff_ff_0_MatMul_pre_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer2_ff_ff_0_MatMul_pre_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__MatMul_301(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__MatMul_301[] = {128, 64};
  VALIDATE(model.addTensor("onnx__MatMul_301", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__MatMul_301",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_onnx__MatMul_301,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__MatMul_301),
                                                .dataSize=BINLEN(onnx__MatMul_301)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_layer2_ff_0_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_layer2_ff_0_bias[] = {128};
  VALIDATE(model.addTensor("layer2_ff_0_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "layer2_ff_0_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_layer2_ff_0_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(layer2_ff_0_bias),
                                                .dataSize=BINLEN(layer2_ff_0_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__layer2_ff_ff_0_MatMul(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_ff_ff_0_MatMul */
  const char*  inputs__layer2_ff_ff_0_MatMul[] = {
    "_layer2_ff_ff_0_MatMul_pre_reshape",
    "onnx__MatMul_301",
    "layer2_ff_0_bias"
  };
  uint32_t dimensions__layer2_ff_ff_0_Add_output_0_fc[] = {8, 128};
  Qnn_Tensor_t outputs__layer2_ff_ff_0_MatMul[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_ff_ff_0_Add_output_0_fc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__layer2_ff_ff_0_Add_output_0_fc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_ff_ff_0_MatMul", // Node Name
                         "qti.aisw", // Package Name
                         "FullyConnected", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer2_ff_ff_0_MatMul, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__layer2_ff_ff_0_MatMul, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_ff_ff_0_MatMul_post_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_ff_ff_0_MatMul_post_reshape */
  const char*  inputs__layer2_ff_ff_0_MatMul_post_reshape[] = {
    "_layer2_ff_ff_0_Add_output_0_fc"
  };
  uint32_t dimensions__layer2_ff_ff_0_Add_output_0[] = {1, 8, 128};
  Qnn_Tensor_t outputs__layer2_ff_ff_0_MatMul_post_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_ff_ff_0_Add_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer2_ff_ff_0_Add_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_ff_ff_0_MatMul_post_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer2_ff_ff_0_MatMul_post_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer2_ff_ff_0_MatMul_post_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__elementwiseneuron_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _elementwiseneuron_3 */
  Qnn_Param_t params__elementwiseneuron_3[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs__elementwiseneuron_3[] = {
    "_layer2_ff_ff_0_Add_output_0"
  };
  uint32_t dimensions__layer2_ff_ff_1_Mul_1_output_0[] = {1, 8, 128};
  Qnn_Tensor_t outputs__elementwiseneuron_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_ff_ff_1_Mul_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer2_ff_ff_1_Mul_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_elementwiseneuron_3", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params__elementwiseneuron_3, // Node Params
                         1, // Num Node Params
                         inputs__elementwiseneuron_3, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__elementwiseneuron_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_ff_ff_2_MatMul_pre_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_ff_ff_2_MatMul_pre_reshape */
  const char*  inputs__layer2_ff_ff_2_MatMul_pre_reshape[] = {
    "_layer2_ff_ff_1_Mul_1_output_0"
  };
  uint32_t dimensions__layer2_ff_ff_2_MatMul_pre_reshape[] = {8, 128};
  Qnn_Tensor_t outputs__layer2_ff_ff_2_MatMul_pre_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_ff_ff_2_MatMul_pre_reshape",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__layer2_ff_ff_2_MatMul_pre_reshape,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_ff_ff_2_MatMul_pre_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer2_ff_ff_2_MatMul_pre_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer2_ff_ff_2_MatMul_pre_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_onnx__MatMul_302(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_onnx__MatMul_302[] = {64, 128};
  VALIDATE(model.addTensor("onnx__MatMul_302", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "onnx__MatMul_302",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_onnx__MatMul_302,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(onnx__MatMul_302),
                                                .dataSize=BINLEN(onnx__MatMul_302)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_layer2_ff_2_bias(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_layer2_ff_2_bias[] = {64};
  VALIDATE(model.addTensor("layer2_ff_2_bias", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "layer2_ff_2_bias",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_layer2_ff_2_bias,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(layer2_ff_2_bias),
                                                .dataSize=BINLEN(layer2_ff_2_bias)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode__layer2_ff_ff_2_MatMul(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_ff_ff_2_MatMul */
  const char*  inputs__layer2_ff_ff_2_MatMul[] = {
    "_layer2_ff_ff_2_MatMul_pre_reshape",
    "onnx__MatMul_302",
    "layer2_ff_2_bias"
  };
  uint32_t dimensions__layer2_ff_ff_2_Add_output_0_fc[] = {8, 64};
  Qnn_Tensor_t outputs__layer2_ff_ff_2_MatMul[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_ff_ff_2_Add_output_0_fc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__layer2_ff_ff_2_Add_output_0_fc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_ff_ff_2_MatMul", // Node Name
                         "qti.aisw", // Package Name
                         "FullyConnected", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer2_ff_ff_2_MatMul, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__layer2_ff_ff_2_MatMul, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_ff_ff_2_MatMul_post_reshape(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_ff_ff_2_MatMul_post_reshape */
  const char*  inputs__layer2_ff_ff_2_MatMul_post_reshape[] = {
    "_layer2_ff_ff_2_Add_output_0_fc"
  };
  uint32_t dimensions__layer2_ff_ff_2_Add_output_0[] = {1, 8, 64};
  Qnn_Tensor_t outputs__layer2_ff_ff_2_MatMul_post_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_ff_ff_2_Add_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer2_ff_ff_2_Add_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_ff_ff_2_MatMul_post_reshape", // Node Name
                         "qti.aisw", // Package Name
                         "Reshape", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__layer2_ff_ff_2_MatMul_post_reshape, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs__layer2_ff_ff_2_MatMul_post_reshape, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_Add_1(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_Add_1 */
  Qnn_Param_t params__layer2_Add_1[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs__layer2_Add_1[] = {
    "_layer2_norm1_LayerNormalization_output_0",
    "_layer2_ff_ff_2_Add_output_0"
  };
  uint32_t dimensions__layer2_Add_1_output_0[] = {1, 8, 64};
  Qnn_Tensor_t outputs__layer2_Add_1[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_Add_1_output_0",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions__layer2_Add_1_output_0,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_Add_1", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params__layer2_Add_1, // Node Params
                         1, // Num Node Params
                         inputs__layer2_Add_1, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs__layer2_Add_1, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode__layer2_norm2_LayerNormalization(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _layer2_norm2_LayerNormalization */
  uint32_t dimensions__layer2_norm2_LayerNormalization_axes[] = {1};
  uint32_t _layer2_norm2_LayerNormalization_axes[] = {2};
  Qnn_Param_t params__layer2_norm2_LayerNormalization[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="axes",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_layer2_norm2_LayerNormalization_axes",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__layer2_norm2_LayerNormalization_axes,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_layer2_norm2_LayerNormalization_axes,
                           .dataSize=4}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="epsilon",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 0.0000099999997473787516355514526367187500f}}}}
  };
  const char*  inputs__layer2_norm2_LayerNormalization[] = {
    "_layer2_Add_1_output_0",
    "layer1_norm1_weight",
    "layer1_attn_out_proj_bias"
  };
  uint32_t dimensions_output[] = {1, 8, 64};
  Qnn_Tensor_t outputs__layer2_norm2_LayerNormalization[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "output",
            .type= QNN_TENSOR_TYPE_APP_READ,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 3,
            .dimensions=dimensions_output,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_layer2_norm2_LayerNormalization", // Node Name
                         "qti.aisw", // Package Name
                         "LayerNorm", // Qnn Node Type
                         params__layer2_norm2_LayerNormalization, // Node Params
                         2, // Num Node Params
                         inputs__layer2_norm2_LayerNormalization, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__layer2_norm2_LayerNormalization, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

QNN_API
ModelError_t QnnModel_composeGraphs(Qnn_BackendHandle_t backendHandle,
                                    QNN_INTERFACE_VER_TYPE interface,
                                    Qnn_ContextHandle_t contextHandle,
                                    const GraphConfigInfo_t** graphsConfigInfo,
                                    const uint32_t numGraphsConfigInfo,
                                    GraphInfoPtr_t** graphsInfo,
                                    uint32_t* numGraphsInfo,
                                    bool debug,
                                    QnnLog_Callback_t logCallback,
                                    QnnLog_Level_t maxLogLevel) {

  ModelError_t err = MODEL_NO_ERROR;

  /* model/graph for transformer2*/
  QnnModel transformer2;
  const QnnGraph_Config_t** graphConfigs = nullptr;
  VALIDATE(getQnnGraphConfigFromInfo("transformer2", graphsConfigInfo, numGraphsConfigInfo, graphConfigs), err);
  VALIDATE(transformer2.initialize(backendHandle, interface, contextHandle, "transformer2", debug, DO_GRAPH_NODE_VALIDATIONS, graphConfigs), err);
  VALIDATE(addTensor_input(transformer2), err);
  VALIDATE(addNode_input_ncf(transformer2), err);
  VALIDATE(addNode__layer1_attn_Transpose(transformer2), err);
  VALIDATE(addTensor_onnx__MatMul_289(transformer2), err);
  VALIDATE(addNode__layer1_attn_MatMul(transformer2), err);
  VALIDATE(addNode__layer1_attn_MatMul_post_reshape(transformer2), err);
  VALIDATE(addNode__layer1_attn_Transpose_1(transformer2), err);
  VALIDATE(addNode__layer1_attn_Squeeze_1(transformer2), err);
  VALIDATE(addTensor__layer1_attn_Constant_1_output_0(transformer2), err);
  VALIDATE(addNode__layer1_attn_Gather_3(transformer2), err);
  VALIDATE(addNode_Reshape_post__layer1_attn_Gather_3(transformer2), err);
  VALIDATE(addTensor__layer1_attn_Constant_output_0(transformer2), err);
  VALIDATE(addNode__layer1_attn_Gather_4(transformer2), err);
  VALIDATE(addNode_Reshape_post__layer1_attn_Gather_4(transformer2), err);
  VALIDATE(addTensor__layer1_attn_Constant_4_output_0(transformer2), err);
  VALIDATE(addNode__layer1_attn_Gather_5(transformer2), err);
  VALIDATE(addNode_Reshape_post__layer1_attn_Gather_5(transformer2), err);
  VALIDATE(addNode__layer1_attn_Transpose_2(transformer2), err);
  VALIDATE(addNode__layer1_attn_Transpose_3(transformer2), err);
  VALIDATE(addTensor__layer1_attn_Constant_24_output_0(transformer2), err);
  VALIDATE(addNode__layer1_attn_Mul_1(transformer2), err);
  VALIDATE(addNode__layer1_attn_Transpose_4(transformer2), err);
  VALIDATE(addNode__layer1_attn_MatMul_1(transformer2), err);
  VALIDATE(addNode__layer1_attn_Softmax(transformer2), err);
  VALIDATE(addNode__layer1_attn_MatMul_2(transformer2), err);
  VALIDATE(addNode__layer1_attn_Transpose_5(transformer2), err);
  VALIDATE(addTensor_layer1_attn_out_proj_weight_permute(transformer2), err);
  VALIDATE(addTensor_layer1_attn_out_proj_bias(transformer2), err);
  VALIDATE(addNode__layer1_attn_Gemm(transformer2), err);
  VALIDATE(addNode__layer1_attn_Reshape_7(transformer2), err);
  VALIDATE(addNode__layer1_attn_Transpose_6(transformer2), err);
  VALIDATE(addNode__layer1_Add(transformer2), err);
  VALIDATE(addTensor_layer1_norm1_weight(transformer2), err);
  VALIDATE(addNode__layer1_norm1_LayerNormalization(transformer2), err);
  VALIDATE(addNode__layer1_ff_ff_0_MatMul_pre_reshape(transformer2), err);
  VALIDATE(addTensor_onnx__MatMul_294(transformer2), err);
  VALIDATE(addTensor_layer1_ff_0_bias(transformer2), err);
  VALIDATE(addNode__layer1_ff_ff_0_MatMul(transformer2), err);
  VALIDATE(addNode__layer1_ff_ff_0_MatMul_post_reshape(transformer2), err);
  VALIDATE(addNode__elementwiseneuron_1(transformer2), err);
  VALIDATE(addNode__layer1_ff_ff_2_MatMul_pre_reshape(transformer2), err);
  VALIDATE(addTensor_onnx__MatMul_295(transformer2), err);
  VALIDATE(addTensor_layer1_ff_2_bias(transformer2), err);
  VALIDATE(addNode__layer1_ff_ff_2_MatMul(transformer2), err);
  VALIDATE(addNode__layer1_ff_ff_2_MatMul_post_reshape(transformer2), err);
  VALIDATE(addNode__layer1_Add_1(transformer2), err);
  VALIDATE(addNode__layer1_norm2_LayerNormalization(transformer2), err);
  VALIDATE(addNode__layer2_attn_Transpose(transformer2), err);
  VALIDATE(addTensor_onnx__MatMul_296(transformer2), err);
  VALIDATE(addNode__layer2_attn_MatMul(transformer2), err);
  VALIDATE(addNode__layer2_attn_MatMul_post_reshape(transformer2), err);
  VALIDATE(addNode__layer2_attn_Transpose_1(transformer2), err);
  VALIDATE(addNode__layer2_attn_Squeeze_1(transformer2), err);
  VALIDATE(addNode__layer2_attn_Gather_3(transformer2), err);
  VALIDATE(addNode_Reshape_post__layer2_attn_Gather_3(transformer2), err);
  VALIDATE(addNode__layer2_attn_Gather_4(transformer2), err);
  VALIDATE(addNode_Reshape_post__layer2_attn_Gather_4(transformer2), err);
  VALIDATE(addNode__layer2_attn_Gather_5(transformer2), err);
  VALIDATE(addNode_Reshape_post__layer2_attn_Gather_5(transformer2), err);
  VALIDATE(addNode__layer2_attn_Transpose_2(transformer2), err);
  VALIDATE(addNode__layer2_attn_Transpose_3(transformer2), err);
  VALIDATE(addNode__layer2_attn_Mul_1(transformer2), err);
  VALIDATE(addNode__layer2_attn_Transpose_4(transformer2), err);
  VALIDATE(addNode__layer2_attn_MatMul_1(transformer2), err);
  VALIDATE(addNode__layer2_attn_Softmax(transformer2), err);
  VALIDATE(addNode__layer2_attn_MatMul_2(transformer2), err);
  VALIDATE(addNode__layer2_attn_Transpose_5(transformer2), err);
  VALIDATE(addTensor_layer2_attn_out_proj_weight_permute(transformer2), err);
  VALIDATE(addNode__layer2_attn_Gemm(transformer2), err);
  VALIDATE(addNode__layer2_attn_Reshape_7(transformer2), err);
  VALIDATE(addNode__layer2_attn_Transpose_6(transformer2), err);
  VALIDATE(addNode__layer2_Add(transformer2), err);
  VALIDATE(addNode__layer2_norm1_LayerNormalization(transformer2), err);
  VALIDATE(addNode__layer2_ff_ff_0_MatMul_pre_reshape(transformer2), err);
  VALIDATE(addTensor_onnx__MatMul_301(transformer2), err);
  VALIDATE(addTensor_layer2_ff_0_bias(transformer2), err);
  VALIDATE(addNode__layer2_ff_ff_0_MatMul(transformer2), err);
  VALIDATE(addNode__layer2_ff_ff_0_MatMul_post_reshape(transformer2), err);
  VALIDATE(addNode__elementwiseneuron_3(transformer2), err);
  VALIDATE(addNode__layer2_ff_ff_2_MatMul_pre_reshape(transformer2), err);
  VALIDATE(addTensor_onnx__MatMul_302(transformer2), err);
  VALIDATE(addTensor_layer2_ff_2_bias(transformer2), err);
  VALIDATE(addNode__layer2_ff_ff_2_MatMul(transformer2), err);
  VALIDATE(addNode__layer2_ff_ff_2_MatMul_post_reshape(transformer2), err);
  VALIDATE(addNode__layer2_Add_1(transformer2), err);
  VALIDATE(addNode__layer2_norm2_LayerNormalization(transformer2), err);

  // Add all models to array to get graphsInfo
  QnnModel* models [] = {&transformer2};
  uint32_t numModels = 1;

  // Populate the constructed graphs in provided output variables
  VALIDATE(getGraphInfoFromModels(*models, numModels, graphsInfo), err);
  *numGraphsInfo = numModels;

  return err;

} // PREPARE_GRAPHS

QNN_API
ModelError_t QnnModel_freeGraphsInfo(GraphInfoPtr_t** graphsInfo, uint32_t numGraphsInfo){
  return qnn_wrapper_api::freeGraphsInfo(graphsInfo, numGraphsInfo);
} // FREEGRAPHINFO

}