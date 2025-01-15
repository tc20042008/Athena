import abstract_drr
import access_topo_drr
import matmul_unary_tpl


@abstract_drr.register_drr_pass("matrix_unary_fusion", nice=0)
class MatrixUnaryFusion(abstract_drr.DrrPass):

  def source_pattern(self, o, t):
    o.matmul_op = o.ap_native_op("pd_op.matmul")
    o.matmul_op(
      [t.input0, t.input1],
      [t.output0]
    )

    o.trivial_op = o.ap_trivial_fusion_op()
    o.trivial_op(
      [t.output0],
      [t.output1]
    )

  def get_constraint_func(self):
    return Constraint.__function__

  def result_pattern(self, o, t):
    o.fustion_op = o.ap_pattern_fusion_op(CodeGen.__function__)
    o.fustion_op(
      [t.input0, t.input1],
      [t.output1]
    )


def Constraint(o, t, ir_helper):
    return True


def CodeGen(ctx, o, t):
  trivial_op_code_gen_class = ctx.make_fusion_op_code_gen_class(
    o.trivial_op,
    input_index_loop_anchor_flags=[False],
    output_index_loop_anchor_flags=[True],
  )
  template_module = matmul_unary_tpl.MatrixAddUnaryTemplate(
    trivial_op_code_gen_class=trivial_op_code_gen_class
  )
  return template_module.compile(
    input0_karg=ctx.in_tensor_data_ptr_kernel_arg_id(t.input0),
    input1_karg=ctx.in_tensor_data_ptr_kernel_arg_id(t.input1),
    output_karg=ctx.out_tensor_data_ptr_kernel_arg_id(t.output1),
    m_karg=ctx.dim_expr_kernel_arg_id(t.input0.shape[0]),
    n_karg=ctx.dim_expr_kernel_arg_id(t.input1.shape[1]),
    k_karg=ctx.dim_expr_kernel_arg_id(t.input0.shape[1]),
  )
