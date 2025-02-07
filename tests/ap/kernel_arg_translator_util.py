class KernelArgTranslator:
  def __init__(self, param_struct_name):
    self.param_struct_name = param_struct_name

  def get_kernel_arg_name(self, var_name):
    return var_name

  def get_param_struct_field_name(self, var_name):
    return var_name

  def get_param_struct_init_name(self, var_name):
    return f"{self.param_struct_name}.{var_name}"

  def get_use_name(self, var_name):
    return f"{self.param_struct_name}.{var_name}"
