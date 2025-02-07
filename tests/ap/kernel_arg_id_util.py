class KernelArgIdNameLazyContext:

  def __init__(self, code_gen_ctx, tensor_match_ctx, name_prefix):
    self.code_gen_ctx = code_gen_ctx
    self.tensor_match_ctx = tensor_match_ctx
    self.name_prefix = name_prefix
    self.kernel_id_arg_id2unique_name = MutableOrderedDict()
    self.in_tensor_data_ptr_seq_no = 0
    self.out_tensor_data_ptr_seq_no = 0
    self.dim_expr_seq_no = 0

  def items(self):
    return self.kernel_id_arg_id2unique_name.items()

  def get_in_tensor_data_ptr_var_name(self, in_ir_value_name):
    ir_value = getattr(self.tensor_match_ctx, in_ir_value_name)
    kernel_arg_id = self.code_gen_ctx.in_tensor_data_ptr_kernel_arg_id(ir_value)
    create = self._create_in_tensor_data_ptr_var_name
    return self.kernel_id_arg_id2unique_name.get_or_create(kernel_arg_id, create)

  def _create_in_tensor_data_ptr_var_name(self):
    name = f"{self.name_prefix}in_ptr_{self.in_tensor_data_ptr_seq_no}"
    self.in_tensor_data_ptr_seq_no = self.in_tensor_data_ptr_seq_no + 1
    return name

  def get_out_tensor_data_ptr_var_name(self, out_ir_value_name):
    ir_value = getattr(self.tensor_match_ctx, out_ir_value_name)
    kernel_arg_id = self.code_gen_ctx.out_tensor_data_ptr_kernel_arg_id(ir_value)
    create = self._create_out_tensor_data_ptr_var_name
    return self.kernel_id_arg_id2unique_name.get_or_create(kernel_arg_id, create)

  def _create_out_tensor_data_ptr_var_name(self):
    name = f"{self.name_prefix}out_ptr_{self.out_tensor_data_ptr_seq_no}"
    self.out_tensor_data_ptr_seq_no = self.out_tensor_data_ptr_seq_no + 1
    return name

  def get_dim_expr_var_name(self, dim_expr):
    kernel_arg_id = self.code_gen_ctx.dim_expr_kernel_arg_id(dim_expr)
    create = self._create_dim_expr_var_name
    return self.kernel_id_arg_id2unique_name.get_or_create(kernel_arg_id, create)

  def _create_dim_expr_var_name(self):
    name = f"{self.name_prefix}dim_{self.dim_expr_seq_no}"
    self.dim_expr_seq_no = self.dim_expr_seq_no + 1
    return name
