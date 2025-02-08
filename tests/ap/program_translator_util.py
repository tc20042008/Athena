class ProgramTranslator:

  def __init__(
    self,
    program_property,
    kernel_arg_translator,
    index_program_translator_map,
    op_translator_maker,
  ):
    self.program_property = program_property
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map
    self.op_translator_maker = op_translator_maker
    self.ir_value_index2translated_value = MutableList()
    def PushNone(x):
      self.ir_value_index2translated_value.append(None)
    map(PushNone, self.program_property.values)

  # mut_kernel_arg_id_lazy_ctx: mutable KernelArgIdLazyContext
  # mut_lir_code_gen_ctx: mutable low level ir code generation context
  def translate(self, mut_kernel_arg_id_lazy_ctx, mut_lir_code_gen_ctx):
    def TranslateOp(op_property):
      self._translate_op(op_property, mut_kernel_arg_id_lazy_ctx, mut_lir_code_gen_ctx)
    map(TranslateOp, self.program_property.ops)

  def _translate_op(self, op_property, mut_kernel_arg_id_lazy_ctx, mut_lir_code_gen_ctx):
    op_translator = self.op_translator_maker(
      op_property=op_property,
      input_properties=map(self._get_value_property, op_property.input_value_indexes),
      output_properties=map(self._get_value_property, op_property.output_value_indexes),
      kernel_arg_translator=self.kernel_arg_translator,
      index_program_translator_map=self.index_program_translator_map,
    )
    inputs = map(self._get_translated_value, op_property.input_value_indexes)
    outputs = op_translator(
      inputs,
      mut_kernel_arg_id_lazy_ctx=mut_kernel_arg_id_lazy_ctx,
      mut_lir_code_gen_ctx=mut_lir_code_gen_ctx
    )
    map(self._set_translated_value, zip(op_property.output_value_indexes, outputs))

  def _get_value_property(self, i):
    return self.program_property.values[i]

  def _get_translated_value(self, i):
    return self.ir_value_index2translated_value[i]

  def _set_translated_value(self, pair):
    self.ir_value_index2translated_value[pair[0]] = pair[1]