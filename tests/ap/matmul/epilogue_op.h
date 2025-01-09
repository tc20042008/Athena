#pragma once

namespace ap {

template <typename T>
struct ScaleFunctor {
  __forceinline__ __host__ __device__
  T operator()(T x) {
    return x * 0.5;
  }
};

}
