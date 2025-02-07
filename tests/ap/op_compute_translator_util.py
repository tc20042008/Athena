import code_gen_value_util

class ApOpLoadFromRegisterCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_lazy_ctx, mut_lir_code_gen_ctx):
    out = self.get_out_cg_val(0)
    return [out]

  def get_out_cg_val(self, i):
    register_var_name_attr = self.op_property.attributes.register_var_name
    register_var_name = register_var_name_attr.match(a_str=lambda x:x)
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      register_var_name
    )


class ApOpLoadFromGlobalCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_lazy_ctx, mut_lir_code_gen_ctx):
    index_func_unique_id_attr = self.op_property.attributes.index_func_unique_id
    index_func_unique_id = index_func_unique_id_attr.match(a_str=lambda x:x)
    offset_var_name = self.index_program_translator_map.get_offset_var_name(
      index_func_unique_id=index_func_unique_id,
      mut_kernel_arg_id_lazy_ctx=mut_kernel_arg_id_lazy_ctx,
      mut_lir_code_gen_ctx=mut_lir_code_gen_ctx,
    )
    data_op_name = inputs[0].var_name
    arg_name = mut_kernel_arg_id_lazy_ctx.get_in_tensor_data_ptr_var_name(data_op_name)
    ptr_var_name = self.kernel_arg_translator.get_use_name(arg_name)
    out = self.get_out_cg_val(0)
    mut_lir_code_gen_ctx.let(out, f"{ptr_var_name}[{offset_var_name}]")
    return [out]

  def get_out_cg_val(self, i):
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      f"op{self.op_property.op_index}_out{i}"
    )


class ApOpStoreToRegisterCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_lazy_ctx, mut_lir_code_gen_ctx):
    mut_lir_code_gen_ctx.stmts.append(f"{self.get_out_var_name()} = {inputs[0].var_name};")
    return []

  def get_out_var_name(self):
    register_var_name_attr = self.op_property.attributes.register_var_name
    return register_var_name_attr.match(a_str=lambda x:x)


class PdOpDataCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_lazy_ctx, mut_lir_code_gen_ctx):
    out = self.get_out_cg_val(0)
    return [out]

  def get_out_cg_val(self, i):
    name = self.op_property.attributes.name.match(a_str=lambda x:x)
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      name
    )


class PdOpExpCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_lazy_ctx, mut_lir_code_gen_ctx):
    out = self.get_out_cg_val(0)
    mut_lir_code_gen_ctx.let(out, f"expf({inputs[0].var_name})")
    return [out]

  def get_out_cg_val(self, i):
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      f"op{self.op_property.op_index}_out{i}"
    )


class PdOpSubstractCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_lazy_ctx, mut_lir_code_gen_ctx):
    a = inputs[0]
    b = inputs[1]
    out = self.get_out_cg_val(0)
    mut_lir_code_gen_ctx.let(out, f"({a.var_name} - {b.var_name})")
    return [out]

  def get_out_cg_val(self, i):
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      f"op{self.op_property.op_index}_out{i}"
    )


class PdOpAddCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_lazy_ctx, mut_lir_code_gen_ctx):
    a = inputs[0]
    b = inputs[1]
    out = self.get_out_cg_val(0)
    mut_lir_code_gen_ctx.let(out, f"({a.var_name} + {b.var_name})")
    return [out]

  def get_out_cg_val(self, i):
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      f"op{self.op_property.op_index}_out{i}"
    )


class PdOpMultiplyCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_lazy_ctx, mut_lir_code_gen_ctx):
    a = inputs[0]
    b = inputs[1]
    out = self.get_out_cg_val(0)
    mut_lir_code_gen_ctx.let(out, f"({a.var_name} * {b.var_name})")
    return [out]

  def get_out_cg_val(self, i):
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      f"op{self.op_property.op_index}_out{i}"
    )


class CinnOpYieldStoreCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_lazy_ctx, mut_lir_code_gen_ctx):
    return inputs


class CinnOpBroadcastCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_lazy_ctx, mut_lir_code_gen_ctx):
    return inputs

class CinnOpGenerateShapeCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_lazy_ctx, mut_lir_code_gen_ctx):
    out = self.get_out_cg_val(0)
    return [out]

  def get_out_cg_val(self, i):
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      f"op{self.op_property.op_index}_out{i}"
    )


class OpComputeTranslatorFactory:
  def __init__(self):
    self.op_name2class = OrderedDict([
      ["ap_op.load_from_register",  ApOpLoadFromRegisterCodeGen],
      ["ap_op.store_to_register",   ApOpStoreToRegisterCodeGen],
      ["ap_op.load_from_global",    ApOpLoadFromGlobalCodeGen],
      ["pd_op.data",                PdOpDataCodeGen],
      ["pd_op.exp",                 PdOpExpCodeGen],
      ["pd_op.subtract",            PdOpSubstractCodeGen],
      ["pd_op.add",                 PdOpAddCodeGen],
      ["pd_op.multiply",            PdOpMultiplyCodeGen],
      ["cinn_op.yield_store",       CinnOpYieldStoreCodeGen],
      ["cinn_op.broadcast",         CinnOpBroadcastCodeGen],
      ["cinn_op.generate_shape",    CinnOpGenerateShapeCodeGen]
    ])

  def __call__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    cls = self._get_class(op_property.op_name)
    return cls(
      op_property=op_property,
      input_properties=input_properties,
      output_properties=output_properties,
      kernel_arg_translator=kernel_arg_translator,
      index_program_translator_map=index_program_translator_map,
    )

  def _get_class(self, op_name):
    return self.op_name2class[op_name]
