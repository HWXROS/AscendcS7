
#include "fmax_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
using namespace optiling;  // 或其他包含 PlatformAscendC 的命名空间
using namespace platform_ascendc;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  FmaxTilingData tiling;
    // 1. 实例化AscendC平台对象（关联当前设备的平台信息）
    // 2. 获取AI Core的UB（Unified Buffer，统一缓冲区）大小
    // 3. 获取当前设备的AI Core总数
    auto platform = PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = platform.GetCoreNum();
    
    if (coreNum == 0) coreNum = 64;       // 默认910B的核心数
    // coreNum=1;
    //--------------------------------------------------------//
    //             判断是否需要广播机制，推导广播结果             //
    //--------------------------------------------------------//
    
    bool all_equal = true;
    bool compatible = true;
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    const gert::StorageShape* x2_shape = context->GetInputShape(1);
    uint32_t x1_dim = x1_shape->GetStorageShape().GetDimNum();
    uint32_t x2_dim = x2_shape->GetStorageShape().GetDimNum();
    uint32_t y_dim=std::max(x1_dim,x2_dim);
    uint32_t x1_newshape[5] = {1,1,1,1,1};
    uint32_t x2_newshape[5] = {1,1,1,1,1};
    uint32_t broadcast_mask_x1[5] = {1,1,1,1,1};
    uint32_t broadcast_mask_x2[5] = {1,1,1,1,1};
    uint32_t y_shape[5] = {1,1,1,1,1};
    uint32_t x1_start = y_dim - x1_dim;
    uint32_t x2_start = y_dim - x2_dim;
    uint32_t y_size=1;
    for (uint32_t i = 0; i < y_dim; i++) {
        x1_newshape[i] = (i < x1_start) ? 1 : x1_shape->GetStorageShape().GetDim(i - x1_start);
        x2_newshape[i] = (i < x2_start) ? 1 : x2_shape->GetStorageShape().GetDim(i - x2_start);
        if (x1_newshape[i] != x2_newshape[i]) {
            all_equal = false;
            if (!(x1_newshape[i] == 1 || x2_newshape[i] == 1)) {
                compatible = false;
                break;
            }
            broadcast_mask_x1[i] = (x1_newshape[i] == 1) ? 0 : 1;
            broadcast_mask_x2[i] = (x2_newshape[i] == 1) ? 0 : 1;
        }
        y_shape[i] = std::max(x1_newshape[i], x2_newshape[i]);
        y_size *= y_shape[i];
    }
    if (all_equal) {
        tiling.set_broadcast(0);
    } else if (compatible) {
        tiling.set_broadcast(1);
    } else {
        return ge::GRAPH_FAILED;
    }
    uint32_t tile_size;
    //  根据类型设置Tile_Size
    ge::DataType in_dtype = context->GetInputDesc(0)->GetDataType();
    switch (in_dtype) {
        case ge::DT_INT64:
            tile_size = 2048;
            tiling.set_Tile_Size(2048);
            break;
        default:
            tile_size = 4096;
            tiling.set_Tile_Size(4096);
            break;
    }

    
    uint32_t block_dim = 0;  // 实际启用的核数
    uint64_t total_blocks = (y_size + tile_size - 1) / tile_size;  // 总块数（向上取整）
    
    // 适配小尺寸：总元素数≤tile_size时，强制单块单核心
    if (total_blocks == 0) total_blocks = 1;
    if (total_blocks <= coreNum) {
        block_dim = static_cast<uint32_t>(total_blocks);  // 小尺寸用少量核
    } else {
        block_dim = coreNum;  // 大尺寸用满所有核
    }
    // 封装Tiling   

    tiling.set_broadcast_mask_x1(broadcast_mask_x1);
    tiling.set_broadcast_mask_x2(broadcast_mask_x2);
    tiling.set_y_shape(y_shape);
    //结尾 操作
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity()
    );
    context->SetBlockDim(block_dim);
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* x2_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    uint32_t x1_dim = x1_shape->GetDimNum();
    uint32_t x2_dim = x2_shape->GetDimNum();
    uint32_t y_dim=std::max(x1_dim,x2_dim);
    uint32_t x1_newshape[5] = {1,1,1,1,1};
    uint32_t x2_newshape[5] = {1,1,1,1,1};
    uint32_t y[5] = {1,1,1,1,1};
    uint32_t x1_start = y_dim - x1_dim;
    uint32_t x2_start = y_dim - x2_dim;
    for (uint32_t i = 0; i < y_dim; i++) {
        x1_newshape[i] = (i < x1_start) ? 1 : x1_shape->GetDim(i - x1_start);
        x2_newshape[i] = (i < x2_start) ? 1 : x2_shape->GetDim(i - x2_start);
        y[i] = std::max(x1_newshape[i], x2_newshape[i]);
    }
    y_shape->SetDimNum(y_dim);
    for (uint32_t i = 0; i < y_dim; i++) {
        y_shape->SetDim(i, y[i]);
    }
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    auto dt0 = context->GetInputDataType(0);
    auto dt1 = context->GetInputDataType(1);
    if (dt0 != dt1) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, dt0);
    return GRAPH_SUCCESS;
}

}


namespace ops {
class Fmax : public OpDef {
public:
    explicit Fmax(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, 
                    ge::DT_FLOAT, 
                    ge::DT_FLOAT16, 
                    ge::DT_INT32, 
                    ge::DT_INT8, 
                    ge::DT_INT64, 
                    ge::DT_INT16, 
                    ge::DT_UINT8})
            .Format({ge::FORMAT_ND,
                    ge::FORMAT_ND, 
                    ge::FORMAT_ND, 
                    ge::FORMAT_ND,
                    ge::FORMAT_ND,
                    ge::FORMAT_ND,
                    ge::FORMAT_ND,
                    ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL,
                 ge::DT_FLOAT, 
                 ge::DT_FLOAT16, 
                 ge::DT_INT32, 
                 ge::DT_INT8,
                  ge::DT_INT64,
                   ge::DT_INT16,
                    ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8, ge::DT_INT64, ge::DT_INT16, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Fmax);
}
