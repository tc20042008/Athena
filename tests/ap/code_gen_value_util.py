
class CodeGenValue:
  def __init__(self, pir_type, var_name):
    self.pir_type = pir_type
    self.var_name = var_name

  def get_dtype(self):
    def convert_to_dtype(pir_dtype, shape, data_layout):
      return pir_dtype.convert_to_dtype()
    print("type(self.pir_type):", type(self.pir_type))
    return self.pir_type.match(t_dtensor=convert_to_dtype)

  def is_dense_tensor_type(self):
    return self.pir_type.get_type_name() == "t_dtensor"
