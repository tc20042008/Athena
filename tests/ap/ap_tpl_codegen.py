

class CodeGenValue:
  def __init__(self, index_tuple_expr, dtype, var_name):
    self.index_tuple_expr = index_tuple_expr
    self.dtype = dtype
    self.var_name = var_name

class IndexExprCodeGen:
  def __init__(self, loop_var_names):
    self.loop_var_names = loop_var_names

class CudaLikeIrCodeGenCtx:
  def __init__(self):
    self.stmts = MutableList()
    self.dtype2type_name = OrderedDict([
      [DataType.float,  "float"],
      [DataType.float16,  "half"],
      [DataType.int32,    "int"],
    ])

  def assign(self, dst, src):
    self.stmts.append(f"{dst.var_name} = {src.var_name};")

  def let(self, dtype, var_name, val_name):
    type_name = self.dtype2type_name[dtype]
    self.stmts.append(f"{type_name} {var_name} = {val_name};")

  def get_stmts_joined_str(self):
    return "\n".join([*self.stmts])


class PdOpExpCodeGen:
  def __init__(self,
               index_expr_code_gen,
               unique_op_name,
               input_dtypes,
               output_dtypes,
               input_index_tuple_exprs,
               output_index_tuple_exprs,
               attrs):
    self.index_expr_code_gen = index_expr_code_gen
    self.unique_op_name = unique_op_name
    self.input_dtypes = input_dtypes
    self.output_dtypes = output_dtypes
    self.input_index_tuple_exprs = input_index_tuple_exprs
    self.output_index_tuple_exprs = output_index_tuple_exprs
    self.attrs = attrs

  def __call__(self, lir_code_gen_ctx, x):
    out = CodeGenValue(
        self.output_index_tuple_exprs[0],
        self.output_dtypes[0],
        f"{self.unique_op_name}_0")
    val = f"expf({x.var_name})"
    lir_code_gen_ctx.let(out.dtype, out.var_name, val)
    return [out]


class PdOpSubstractCodeGen:
  def __init__(self,
               index_expr_code_gen,
               unique_op_name,
               input_dtypes,
               output_dtypes,
               input_index_tuple_exprs,
               output_index_tuple_exprs,
               attrs):
    self.index_expr_code_gen = index_expr_code_gen
    self.unique_op_name = unique_op_name
    self.input_dtypes = input_dtypes
    self.output_dtypes = output_dtypes
    self.input_index_tuple_exprs = input_index_tuple_exprs
    self.output_index_tuple_exprs = output_index_tuple_exprs
    self.attrs = attrs

  def __call__(self, lir_code_gen_ctx, a, b):
    out = CodeGenValue(
        self.output_index_tuple_exprs[0],
        self.output_dtypes[0],
        f"{self.unique_op_name}_0")
    val = f"({a.var_name} - {b.var_name})"
    lir_code_gen_ctx.let(out.dtype, out.var_name, val)
    return [out]


class PdOpAddCodeGen:
  def __init__(self,
               index_expr_code_gen,
               unique_op_name,
               input_dtypes,
               output_dtypes,
               input_index_tuple_exprs,
               output_index_tuple_exprs,
               attrs):
    self.index_expr_code_gen = index_expr_code_gen
    self.unique_op_name = unique_op_name
    self.input_dtypes = input_dtypes
    self.output_dtypes = output_dtypes
    self.input_index_tuple_exprs = input_index_tuple_exprs
    self.output_index_tuple_exprs = output_index_tuple_exprs
    self.attrs = attrs

  def __call__(self, lir_code_gen_ctx, a, b):
    out = CodeGenValue(
        self.output_index_tuple_exprs[0],
        self.output_dtypes[0],
        f"{self.unique_op_name}_0")
    val = f"({a.var_name} + {b.var_name})"
    lir_code_gen_ctx.let(out.dtype, out.var_name, val)
    return [out]


class PdOpMultiplyCodeGen:
  def __init__(self,
               index_expr_code_gen,
               unique_op_name,
               input_dtypes,
               output_dtypes,
               input_index_tuple_exprs,
               output_index_tuple_exprs,
               attrs):
    self.index_expr_code_gen = index_expr_code_gen
    self.unique_op_name = unique_op_name
    self.input_dtypes = input_dtypes
    self.output_dtypes = output_dtypes
    self.input_index_tuple_exprs = input_index_tuple_exprs
    self.output_index_tuple_exprs = output_index_tuple_exprs
    self.attrs = attrs

  def __call__(self, lir_code_gen_ctx, a, b):
    out = CodeGenValue(
        self.output_index_tuple_exprs[0],
        self.output_dtypes[0],
        f"{self.unique_op_name}_0")
    val = f"({a.var_name} * {b.var_name})"
    lir_code_gen_ctx.let(out.dtype, out.var_name, val)
    return [out]


class CinnOpYieldStoreCodeGen:
  def __init__(self,
               index_expr_code_gen,
               unique_op_name,
               input_dtypes,
               output_dtypes,
               input_index_tuple_exprs,
               output_index_tuple_exprs,
               attrs):
    self.index_expr_code_gen = index_expr_code_gen
    self.unique_op_name = unique_op_name
    self.input_dtypes = input_dtypes
    self.output_dtypes = output_dtypes
    self.input_index_tuple_exprs = input_index_tuple_exprs
    self.output_index_tuple_exprs = output_index_tuple_exprs
    self.attrs = attrs

  def __call__(self, lir_code_gen_ctx, x):
    return [x]


class CinnOpBroadcastCodeGen:
  def __init__(self,
               index_expr_code_gen,
               unique_op_name,
               input_dtypes,
               output_dtypes,
               input_index_tuple_exprs,
               output_index_tuple_exprs,
               attrs):
    self.index_expr_code_gen = index_expr_code_gen
    self.unique_op_name = unique_op_name
    self.input_dtypes = input_dtypes
    self.output_dtypes = output_dtypes
    self.input_index_tuple_exprs = input_index_tuple_exprs
    self.output_index_tuple_exprs = output_index_tuple_exprs
    self.attrs = attrs

  def __call__(self, lir_code_gen_ctx, x):
    return [x]


class CinnOpGenerateShapeCodeGen:
  def __init__(self,
               index_expr_code_gen,
               unique_op_name,
               input_dtypes,
               output_dtypes,
               input_index_tuple_exprs,
               output_index_tuple_exprs,
               attrs):
    self.index_expr_code_gen = index_expr_code_gen
    self.unique_op_name = unique_op_name
    self.input_dtypes = input_dtypes
    self.output_dtypes = output_dtypes
    self.input_index_tuple_exprs = input_index_tuple_exprs
    self.output_index_tuple_exprs = output_index_tuple_exprs
    self.attrs = attrs

  def __call__(self, lir_code_gen_ctx, x):
    out = CodeGenValue(
        self.output_index_tuple_exprs[0],
        self.output_dtypes[0],
        f"{self.unique_op_name}_0")
    return [out]


class NativeOpCodeGenClassFactory:
  def __init__(self):
    self.op_name2class = OrderedDict([
      ["pd_op.exp",               PdOpExpCodeGen],
      ["pd_op.subtract",          PdOpSubstractCodeGen],
      ["pd_op.add",               PdOpAddCodeGen],
      ["pd_op.multiply",          PdOpMultiplyCodeGen],
      ["cinn_op.yield_store",     CinnOpYieldStoreCodeGen],
      ["cinn_op.broadcast",       CinnOpBroadcastCodeGen],
      ["cinn_op.generate_shape",  CinnOpGenerateShapeCodeGen]
    ])

  def __call__(self, op_name):
    return self.op_name2class[op_name]

class GeneratorClassFactory:
  def __init__(self):
    self.native_op_code_gen_class_factory = NativeOpCodeGenClassFactory()

  def get_index_expr_code_generator_class(self):
    return IndexExprCodeGen

  def get_native_op_code_generator_class(self):
    return self.native_op_code_gen_class_factory

  def get_value_class(self):
    return CodeGenValue
