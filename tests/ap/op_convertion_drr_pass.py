import access_topo_drr


def ReturnTrue(o, t, ir_helper):
  return True


@access_topo_drr.register_drr_pass("pd_op_exp", tag="default")
class PdOpExpAccessTopoPass(access_topo_drr.DrrPass):

  def get_constraint_func(self):
    return ReturnTrue.__function__

  def source_pattern(self, o, t):
    o.exp_op = o.ap_native_op("pd_op.exp")
    o.exp_op(
      [t.input],
      [t.output]
    )

  def result_pattern(self, o, t):
    o.fustion_op = o.ap_native_op("pd_op.relu")
    o.fustion_op(
      [t.input],
      [t.output]
    )

@access_topo_drr.register_drr_pass("pd_op_sin", tag="default")
class PdOpSinAccessTopoPass(access_topo_drr.DrrPass):

  def get_constraint_func(self):
    return ReturnTrue.__function__

  def source_pattern(self, o, t):
    o.sin_op = o.ap_native_op("pd_op.sin")
    o.sin_op(
      [t.input],
      [t.output]
    )

  def result_pattern(self, o, t):
    o.fustion_op = o.ap_native_op("pd_op.relu")
    o.fustion_op(
      [t.input],
      [t.output]
    )

@access_topo_drr.register_drr_pass("cinn_op_yield_store", tag="default")
class CinnOpYieldStoreAccessTopoPass(access_topo_drr.DrrPass):

  def get_constraint_func(self):
    return ReturnTrue.__function__

  def source_pattern(self, o, t):
    o.exp_op = o.ap_native_op("cinn_op.yield_store")
    o.exp_op(
      [t.input],
      [t.output]
    )

  def result_pattern(self, o, t):
    o.fustion_op = o.ap_native_op("pd_op.relu")
    o.fustion_op(
      [t.input],
      [t.output]
    )

@access_topo_drr.register_drr_pass("pd_op_subtract", tag="default")
class PdOpSubtractAccessTopoPass(access_topo_drr.DrrPass):

  def get_constraint_func(self):
    return ReturnTrue.__function__

  def source_pattern(self, o, t):
    o.source_op = o.ap_native_op("pd_op.subtract")
    o.source_op(
      [t.input0, t.input1],
      [t.output]
    )

  def result_pattern(self, o, t):
    o.result_op = o.ap_native_op("pd_op.add")
    o.result_op(
      [t.input0, t.input1],
      [t.output]
    )
