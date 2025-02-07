import access_topo_drr

class CutDataPairAccessTopoPass(access_topo_drr.DrrPass):

  def __init__(self, src_data_name, dst_data_name):
    self.src_data_name = pir.a_str(src_data_name)
    self.dst_data_name = pir.a_str(dst_data_name)

  def source_pattern(self, o, t):
    o.data_op = o.ap_native_op("pd_op.data")
    o.data_op(
      [],
      [t.output]
    )

  def constraint(self, o, t):
    return o.data_op.name == self.data_input_name_attr

  def result_pattern(self, o, t):
    t.declare_internal_native_ir_value("input")
    o.new_data_op = o.ap_native_op("pd_op.data")
    o.new_data_op.name = lambda o, t: o.data_op.name
    o.new_data_op.shape = lambda o, t: o.data_op.shape
    o.new_data_op.dtype = lambda o, t: o.data_op.dtype
    o.new_data_op.place = lambda o, t: o.data_op.place
    o.new_data_op(
      [],
      [t.input]
    )
    o.down_spider = o.ap_native_op("ap_op.down_spider")
    o.down_spider(
      [t.input],
      [t.output]
    )

