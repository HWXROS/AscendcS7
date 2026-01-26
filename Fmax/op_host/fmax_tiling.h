
#include "register/tilingdata_base.h"
//  输入维度范围：N∈[1,10000]、N2∈[1,10000]、N3∈[1,10000]、N4∈[1,200]和未知的可能存在的Batchsize
// N4 可能为非 32 的整倍数，需考虑非对齐场景；数据取值范围不超出对应数据类型的表达范围
namespace optiling {
BEGIN_TILING_DATA_DEF(FmaxTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, broadcast);//是否广播 0：否 1：是
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 5,broadcast_mask_x1);//广播mask
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 5,broadcast_mask_x2);//广播mask
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, y_shape);
  TILING_DATA_FIELD_DEF(uint32_t, Tile_Size);//每个Tile处理的元素数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Fmax, FmaxTilingData)
}
