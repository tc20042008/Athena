import abstract_drr
import matmul_unary_tpl


@abstract_drr.register_drr_pass("matmul_unary_fusion", nice=0)
class MatmulUnaryFusion(abstract_drr.DrrPass):
    def source_pattern(self, o, t):
        o.matmul_op = o.ap_native_op("pd_op.matmul")
        o.matmul_op([t.input0, t.input1], [t.output0])

        o.trivial_op = o.ap_trivial_fusion_op()
        o.trivial_op([t.output0], [t.output1])

    def get_constraint_func(self):
        return Constraint.__function__

    def result_pattern(self, o, t):
        o.fustion_op = o.ap_pattern_fusion_op(CodeGen.__function__)
        o.fustion_op([t.input0, t.input1], [t.output1])


def DataDownSpiderYield(o, t):
    o.data_op = o.ap_native_op("pd_op.data")
    o.data_op([], [t.data_op_out])
    o.id_down_spider_op = o.ap_native_op("ap_op.id_down_spider")
    o.id_down_spider_op([t.data_op_out], [t.id_down_spider_op_out])
    o.yield_op = o.ap_native_op("cf.yield")
    o.yield_op([t.id_down_spider_op_out], [])


def Constraint(o, t, ir_helper):
    program = ir_helper.copy_fused_ops_to_program(o.trivial_op)
    print("before-access_topo_pass", program)
    init_pass_manager = ir_helper.create_pass_manager()
    init_pass_manager.add_pass(
        ir_helper.create_access_topo_drr_one_step_pass("init_id_down_spider")
    )
    init_pass_manager.run(program)
    print("after-init-access_topo_pass", program)
    pass_manager = ir_helper.create_pass_manager()
    pass_manager.add_pass(ir_helper.create_access_topo_drr_pass("default"))
    pass_manager.add_pass(ir_helper.create_dce_pass())
    pass_manager.run(program)
    print("after-apply-access_topo_pass", program)
    matched = ir_helper.match(program, DataDownSpiderYield.__function__)
    print("DataDownSpiderYield matched: ", matched)
    return matched


def CodeGen(ctx, o, t):
    trivial_op_code_gen_class = ctx.make_fusion_op_code_gen_class(
        o.trivial_op,
        input_index_loop_anchor_flags=[False],
        output_index_loop_anchor_flags=[True],
    )
    template_module = matmul_unary_tpl.MatmulUnaryTemplate(
        trivial_op_code_gen_class=trivial_op_code_gen_class
    )
    dtypes = [t.input0.dtype, t.input1.dtype, t.output1.dtype]
    return template_module.compile(
        dtype_of_ptr_args=dtypes,
        input0_karg=ctx.in_tensor_data_ptr_kernel_arg_id(t.input0),
        input1_karg=ctx.in_tensor_data_ptr_kernel_arg_id(t.input1),
        output_karg=ctx.out_tensor_data_ptr_kernel_arg_id(t.output1),
        batch_count_karg=ctx.dim_expr_kernel_arg_id(t.input0.shape[0]),
        m_karg=ctx.dim_expr_kernel_arg_id(t.input0.shape[1]),
        n_karg=ctx.dim_expr_kernel_arg_id(t.input1.shape[1]),
        k_karg=ctx.dim_expr_kernel_arg_id(t.input0.shape[2]),
    )
