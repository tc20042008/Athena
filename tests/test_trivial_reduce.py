import abstract_drr
import trivial_reduce_tpl

@abstract_drr.register_drr_pass("trivial_fusion", nice=0)
class TrivialFusionDemo(abstract_drr.DrrPass):

  def source_pattern(self, o, t):
    o.trivial_op = o.ap_trivial_fusion_op()
    o.trivial_op(
      [t.input0],
      [t.output0]
    )
    o.full_int_array = o.ap_native_op("pd_op.full_int_array")
    o.full_int_array(
      [],
      [t.axis]
    )
    o.reduce_op = o.ap_native_op("pd_op.sum")
    o.reduce_op(
      [t.output0, t.axis],
      [t.output1]
    )

  def get_constraint_func(self):
    return Constraint.__function__

  def result_pattern(self, o, t):
    o.fustion_op = o.ap_pattern_fusion_op(CodeGen.__function__)
    o.fustion_op(
      [t.input0],
      [t.output1]
    )

def Constraint(o, t):
  return True

def CodeGen(ctx, o, t):
  trivial_op_code_gen_class = ctx.make_fusion_op_code_gen_class(
    o.trivial_op,
    input_index_loop_anchor_flags=[False],
    output_index_loop_anchor_flags=[True],
  )
  template_module = trivial_reduce_tpl.TrivialReduceTemplate(
    trivial_op_code_gen_class=trivial_op_code_gen_class
  )
  return template_module.compile(
    dim0_karg=ctx.dim_expr_kernel_arg_id(t.input0.shape[0]),
    input_karg=ctx.in_tensor_data_ptr_kernel_arg_id(t.input0),
    output_karg=ctx.out_tensor_data_ptr_kernel_arg_id(t.output1),
  )
