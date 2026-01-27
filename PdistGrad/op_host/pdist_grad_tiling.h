
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PdistGradTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, n);
  TILING_DATA_FIELD_DEF(uint32_t, d);
  TILING_DATA_FIELD_DEF(uint32_t, totalDist);
  TILING_DATA_FIELD_DEF(float, p);
  TILING_DATA_FIELD_DEF(uint32_t, blockSize);
  TILING_DATA_FIELD_DEF(uint32_t, blockNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PdistGrad, PdistGradTilingData)
}
