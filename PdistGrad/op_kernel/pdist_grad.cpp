#include "kernel_operator.h"

extern "C" __global__ __aicore__ void pdist_grad(GM_ADDR grad, GM_ADDR input, GM_ADDR pdist_ouput, GM_ADDR result, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}