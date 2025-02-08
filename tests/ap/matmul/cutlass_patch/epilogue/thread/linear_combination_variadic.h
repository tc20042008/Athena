/*! \file
  \brief Functor performing linear combination operations used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/scale_type.h"


namespace cutlass {
namespace epilogue {
namespace thread {


template <class VariadicOp, class = void>
struct GenericVariadicTraits {
  static constexpr bool IsArgumentsNeeded = false;
  struct Arguments {};
};

template <class VariadicOp>
struct GenericVariadicTraits<VariadicOp, decltype(typename VariadicOp::Arguments(), void())> {
  static constexpr bool IsArgumentsNeeded = true;
  using Arguments = typename VariadicOp::Arguments;
};

/// Applies a linear combination operator to an array of elements.
///
/// D = VariadicOp(alpha * accumulator + beta * source)
///
template <
  template<typename T> class VariadicOp,
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int ElementsPerAccess,                               ///< Number of elements computed per operation.
                                                       ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                                                       ///< but we use 64 or 32 sometimes when there are not enough data to store
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  ScaleType::Kind Scale = ScaleType::Default,          ///< Control Alpha and Beta scaling
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
  bool IsHeavy = false
>
class LinearCombinationVariadic {
public:

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using VariadicArguments = typename GenericVariadicTraits<VariadicOp<ElementCompute>>::Arguments;

  static bool const kIsHeavy = IsHeavy;
  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kCount = ElementsPerAccess;
  static const ScaleType::Kind kScale = Scale;

  using FragmentOutput = Array<ElementOutput, kElementsPerAccess>;
  using FragmentAccumulator = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentSource = Array<ElementOutput, kElementsPerAccess>;
  using FragmentCompute = Array<ElementCompute, kElementsPerAccess>;

  static FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params
  {
    ElementCompute alpha;                         ///< scales accumulators
    ElementCompute beta;                          ///< scales source tensor
    ElementCompute const *alpha_ptr;              ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const *beta_ptr;               ///< pointer to source scalar - if not null, loads it from memory
    VariadicArguments variadic_args;

    CUTLASS_HOST_DEVICE
    Params():
      alpha(ElementCompute(1)),
      beta(ElementCompute(0)),
      alpha_ptr(nullptr),
      beta_ptr(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha,
      ElementCompute beta,
      VariadicArguments variadic_args_ = VariadicArguments{}
    ) : alpha(alpha), beta(beta), alpha_ptr(nullptr), beta_ptr(nullptr), variadic_args(variadic_args_) {}
  };

private:

  //
  // Data members
  //

  Params params_;
  bool skip_elementwise_;

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  LinearCombinationVariadic(Params const &params) {
    params_ = params;
    params_.alpha = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    params_.beta = (params.beta_ptr ? *params.beta_ptr : params.beta);
    skip_elementwise_ = false;
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    if (Scale == ScaleType::NoBetaScaling) return true;

    if (Scale == ScaleType::OnlyAlphaScaling) return false;

    if (Scale == ScaleType::Nothing) return false;

    return params_.beta != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      params_.beta = ElementCompute(1);
    }

    if (k_partition != k_partition_count - 1) {
      skip_elementwise_ = true;
    }
  }

  /// Computes linear scaling with source: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const &accumulator,
      FragmentSource const &source,
      int row_offset,
      int column_offset) const {

    // Convert source to internal compute numeric type
    NumericArrayConverter<ElementCompute, ElementOutput, kElementsPerAccess, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess, Round> accumulator_converter;

    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    FragmentCompute intermediate;

    multiplies<FragmentCompute> mul_add_source;
    multiply_add<FragmentCompute> mul_add_accumulator;
    VariadicOp<ElementCompute> variadic_op;

    if (Scale == ScaleType::NoBetaScaling) {
      intermediate = converted_source;
      intermediate = mul_add_accumulator(params_.alpha, converted_accumulator, intermediate); // D = alpha * Accum + X
    } else if (Scale == ScaleType::Nothing) {
      intermediate = converted_accumulator;
    } else {
      intermediate = mul_add_source(params_.beta, converted_source);                          // X =  beta * C + uniform
      intermediate = mul_add_accumulator(params_.alpha, converted_accumulator, intermediate); // D = alpha * Accum + X
    }

    if constexpr (GenericVariadicTraits<VariadicOp<ElementCompute>>::IsArgumentsNeeded) {
      if (!skip_elementwise_) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i) {
          intermediate[i] = variadic_op(intermediate[i], params_.variadic_args, row_offset, column_offset + i);
        }
      }
    } else {
      if (!skip_elementwise_) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i) {
          intermediate[i] = variadic_op(intermediate[i]);
        }
      }
    }

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kElementsPerAccess, Round> destination_converter;

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = alpha * accumulator
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const &accumulator, int row_offset, int column_offset) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess, Round> accumulator_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    FragmentCompute intermediate;

    multiplies<FragmentCompute> mul_accumulator;
    VariadicOp<ElementCompute> variadic_op;

    if (Scale == ScaleType::Nothing) {
      intermediate = converted_accumulator;
    } else {
      intermediate = mul_accumulator(params_.alpha, converted_accumulator);    // D = alpha * Accum
    }

    if constexpr (GenericVariadicTraits<VariadicOp<FragmentCompute>>::IsArgumentsNeeded) {
      if (!skip_elementwise_) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i) {
          intermediate[i] = variadic_op(intermediate[i], params_.variadic_args, row_offset, column_offset + i);
        }
      }
    } else {
      if (!skip_elementwise_) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i) {
          intermediate[i] = variadic_op(intermediate[i]);
        }
      }
    }

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    return destination_converter(intermediate);
  }
};


} // namespace thread
} // namespace epilogue
} // namespace cutlass
