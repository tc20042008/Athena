import abstract_drr
import topo_drr_pass
import op_convertion_drr_pass
import trivial_reduce_tpl
import ir_tools

def DataDownSpiderYield(o, t):
  o.data_op = o.ap_native_op("pd_op.data")
  o.data_op(
    [],
    [t.data_op_out]
  )
  o.down_spider_op = o.ap_native_op("ap_op.down_spider")
  o.down_spider_op(
    [t.data_op_out],
    [t.down_spider_op_out]
  )
  o.yield_op = o.ap_native_op("cf.yield")
  o.yield_op(
    [t.down_spider_op_out],
    []
  )

@abstract_drr.register_drr_pass("binary_trivial_fusion", nice=0)
class BinaryTrivialFusionDemo(abstract_drr.DrrPass):

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

  def constraint(self, o, t):
    program = ir_tools.copy_fused_ops_to_program(o.trivial_op, tensor_match_ctx=t)
    print("before-access_topo_pass", program)
    init_pass_manager = ir_tools.create_pass_manager()
    init_down_spider = topo_drr_pass.InitDownSpiderAccessTopoPass("in0")
    init_pass_manager.add_pass(
        ir_tools.create_access_topo_drr_one_step_pass(init_down_spider)
    )
    init_pass_manager.run(program)
    print("after-init-access_topo_pass", program)
    pass_manager = ir_tools.create_pass_manager()
    pass_manager.add_pass(ir_tools.create_access_topo_drr_pass("default"))
    pass_manager.add_pass(ir_tools.create_dce_pass())
    pass_manager.run(program)
    print("after-apply-access_topo_pass", program)
    matched = ir_tools.match(program, DataDownSpiderYield.__function__)
    print("DataDownSpiderYield matched: ", matched)
    return matched

  def result_pattern(self, o, t):
    o.fustion_op = o.ap_pattern_fusion_op(self.code_gen)
    o.fustion_op(
      [t.input0],
      [t.output1]
    )

  def code_gen(self, ctx, o, t):
    trivial_op_code_gen_class = ctx.make_fusion_op_code_gen_class(
      o.trivial_op,
      input_index_loop_anchor_flags=[False],
      output_index_loop_anchor_flags=[True],
    )
    template_module = trivial_reduce_tpl.TrivialReduceTemplate(
      trivial_op_code_gen_class=trivial_op_code_gen_class
    )
    return template_module.compile(
      dim0_karg=ctx.dim_expr_kernel_arg_id(t.input0.symbolic_shape_to_list()[0]),
      input_karg=ctx.in_tensor_data_ptr_kernel_arg_id(t.input0),
      output_karg=ctx.out_tensor_data_ptr_kernel_arg_id(t.output1),
    )
