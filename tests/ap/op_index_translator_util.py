import index_code_gen_value_util

class PdOpDataCodeGen:
  def __init__(self,
               index_program_id,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               loop_iter_var_names):
    self.index_program_id = index_program_id
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.loop_iter_var_names = loop_iter_var_names

  def __call__(self, inputs, mut_kernel_arg_id_lazy_ctx, mut_lir_code_gen_ctx):
    return [index_code_gen_value_util.IndexCodeGenValue(self.loop_iter_var_names)]

class CinnOpReshapeCodeGen:
  def __init__(self,
               index_program_id,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               loop_iter_var_names):
    self.index_program_id = index_program_id
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.loop_iter_var_names = loop_iter_var_names

  def __call__(self, inputs, mut_kernel_arg_id_lazy_ctx, mut_lir_code_gen_ctx):
    symbolic_shape = self.input_properties[0].symbolic_shape
    def get_or_create_dim_var_name(dim_expr):
      arg_var_name = mut_kernel_arg_id_lazy_ctx.get_dim_expr_var_name(dim_expr)
      return self.kernel_arg_translator.get_use_name(arg_var_name)
    def get_dim_var_name(i):
      dim_expr = symbolic_shape[i]
      ret = dim_expr.match(
        int64=lambda dim: f"{dim}",
        symbol=lambda sym: get_or_create_dim_var_name(dim_expr),
        _=lambda: get_or_create_dim_var_name(dim_expr)
      )
      return ret
    rank = len(symbolic_shape)
    stride_dims_list = map(
      lambda num_dims: map(lambda i: get_dim_var_name(i + 1), range(rank - 1 - num_dims)),
      range(rank)
    )
    var_name_and_dims_list = map(
      lambda pair: [pair[0], *pair[1]],
      zip(inputs[0].iter_var_names, stride_dims_list)
    )
    offset_expr = reduce(
      lambda elt, acc: f"{acc} + {elt}",
      map(
        lambda elts: reduce(lambda elt, acc: f"{acc} * {elt}", elts),
        var_name_and_dims_list
      )
    )
    # assert len(self.output_properties[0]) == 1
    return [index_code_gen_value_util.IndexCodeGenValue([f"({offset_expr})"])]


class CfYieldCodeGen:
  def __init__(self,
               index_program_id,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               loop_iter_var_names):
    self.index_program_id = index_program_id
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.loop_iter_var_names = loop_iter_var_names

  def __call__(self, inputs, mut_kernel_arg_id_lazy_ctx, mut_lir_code_gen_ctx):
    return []



class OpIndexTranslatorFactory:
  def __init__(self):
    self.op_name2class = OrderedDict([
      ["pd_op.data",                PdOpDataCodeGen],
      ["cinn_op.reshape",           CinnOpReshapeCodeGen],
      ["cf.yield",                  CfYieldCodeGen],
    ])

  def __call__(self,
               index_program_id,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               loop_iter_var_names):
    cls = self._get_class(op_property.op_name)
    return cls(
      index_program_id=index_program_id,
      op_property=op_property,
      input_properties=input_properties,
      output_properties=output_properties,
      kernel_arg_translator=kernel_arg_translator,
      loop_iter_var_names=loop_iter_var_names,
    )

  def _get_class(self, op_name):
    return self.op_name2class[op_name]
