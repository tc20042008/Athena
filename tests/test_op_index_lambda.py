RegisterOpIndexLambda(
    ["pd_op.full_int_array"],
    lambda: lambda: lambda out_shape0: lambda i: MakeTensorIndexes(
        InputTensorIndexes(), OutputTensorIndexes(kIntArrayLikeIndexes)
    ),
)

RegisterOpIndexLambda(
    ["pd_op.reduce"],
    lambda x_shape, axis_shape: lambda x_data, axis_data: lambda y_shape: lambda i: MakeTensorIndexes(
        InputTensorIndexes(kUnsupported, kIntArrayLikeIndexes),
        OutputTensorIndexes(out_data0),
    ),
)


RegisterOpIndexLambda(
    ["pd_op.reshape"],
    lambda x_shape, s_shape: lambda x_data, s_data: lambda y_shape, xs_shape: lambda index: MakeTensorIndexes(
        InputTensorIndexes(
            IndexUndot(IndexDot(index, out_shape0), in_shape0), kIntArrayLikeIndexes
        ),
        OutputTensorIndexes(index, kNothing),
    ),
)
