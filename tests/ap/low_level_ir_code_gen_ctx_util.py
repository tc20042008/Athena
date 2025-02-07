class CudaLikeIrCodeGenCtx:
  def __init__(self):
    self.stmts = MutableList()
    self.dtype2type_name = OrderedDict([
      [DataType.float,  "float"],
      [DataType.int32,    "int"],
    ])

  def assign(self, dst, src):
    self.stmts.append(f"{dst.var_name} = {src.var_name};")

  def let(self, var, val_name):
    type_name = self.dtype2type_name[var.get_dtype()]
    self.stmts.append(f"{type_name} {var.var_name} = {val_name};")

  def get_stmts_joined_str(self):
    return "\n".join([*self.stmts])

