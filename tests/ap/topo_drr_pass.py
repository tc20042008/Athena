import access_topo_drr


def ReturnTrue(o, t, ir_helper):
  return True


@access_topo_drr.register_drr_pass("init_id_down_spider", tag="init_id_down_spider")
class InitIdDownSpiderAccessTopoPass(access_topo_drr.DrrPass):

  def get_constraint_func(self):
    return ReturnTrue.__function__

  def source_pattern(self, o, t):
    o.data_op = o.ap_native_op("pd_op.data")
    o.data_op(
      [],
      [t.output]
    )

  def result_pattern(self, o, t):
    t.declare_internal_native_ir_value("input")
    o.new_data_op = o.ap_native_op("pd_op.data")
    def NameGetter(o, t):
      return o.data_op.name
    o.new_data_op.name = NameGetter.__function__
    def ShapeGetter(o, t):
      return o.data_op.shape
    o.new_data_op.shape = ShapeGetter.__function__
    def DtypeGetter(o, t):
      return o.data_op.dtype
    o.new_data_op.dtype = DtypeGetter.__function__
    def PlaceGetter(o, t):
      return o.data_op.place
    o.new_data_op.place = PlaceGetter.__function__
    o.new_data_op(
      [],
      [t.input]
    )
    o.id_down_spider = o.ap_native_op("ap_op.id_down_spider")
    o.id_down_spider(
      [t.input],
      [t.output]
    )

@access_topo_drr.register_drr_pass("down_spider_relu", tag="default")
class DoubleReluAccessTopoPass(access_topo_drr.DrrPass):

  def get_constraint_func(self):
    return ReturnTrue.__function__

  def source_pattern(self, o, t):
    o.spider0 = o.ap_native_op("ap_op.id_down_spider")
    o.spider0(
      [t.input],
      [t.tmp]
    )
    o.relu1 = o.ap_native_op("pd_op.relu")
    o.relu1(
      [t.tmp],
      [t.output]
    )

  def result_pattern(self, o, t):
    o.fustion_op = o.ap_native_op("ap_op.id_down_spider")
    o.fustion_op(
      [t.input],
      [t.output]
    )


@access_topo_drr.register_drr_pass("down_spider_add", tag="default")
class DoubleReluAccessTopoPass(access_topo_drr.DrrPass):

  def get_constraint_func(self):
    return ReturnTrue.__function__

  def source_pattern(self, o, t):
    o.spider0 = o.ap_native_op("ap_op.id_down_spider")
    o.spider0(
      [t.input],
      [t.tmp0]
    )
    o.spider1 = o.ap_native_op("ap_op.id_down_spider")
    o.spider1(
      [t.input],
      [t.tmp1]
    )
    o.add = o.ap_native_op("pd_op.add")
    o.add(
      [t.tmp0, t.tmp1],
      [t.output]
    )

  def result_pattern(self, o, t):
    o.fustion_op = o.ap_native_op("ap_op.id_down_spider")
    o.fustion_op(
      [t.input],
      [t.output]
    )
        