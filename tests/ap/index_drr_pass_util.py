import access_topo_drr
import pir

class InsertReshapeBeforeYieldPass(access_topo_drr.DrrPass):

  def source_pattern(self, o, t):
    o.yield_op = o.ap_native_op("cf.yield")
    o.yield_op(
      [t.output],
      []
    )

  def result_pattern(self, o, t):
    t.declare_internal_native_ir_value("reshaped_output")
    o.reshape_op = o.ap_native_op("cinn_op.reshape")
    o.reshape_op.shape = lambda o, t: pir.a_array([pir.a_i32(DataValue.int32("-1"))])
    o.reshape_op(
      [t.output],
      [t.reshaped_output]
    )
    o.yield_op = o.ap_native_op("cf.yield")
    o.yield_op(
      [t.reshaped_output],
      []
    )
  