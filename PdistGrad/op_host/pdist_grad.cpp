
#include "pdist_grad_tiling.h"
#include "register/op_def_registry.h"


<<<<<<< HEAD

//  自动选择分块大小
inline uint32_t GetOptimalBlockSize(uint32_t d) {
    if (d <= 256) return d;        // 小数据，一次性处理
    else if (d <= 1024) return 256; // 中等数据
    else return 512;               // 大数据，d可达10000
}


=======
>>>>>>> 41a96658ee2cc44167b5b073209842e137ceacde
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  PdistGradTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
<<<<<<< HEAD
  float p = *context->GetAttrs()->GetFloat(0); // 0: euclidean, 1: cityblock 2

 uint32_t n = x1_shape->GetStorageShape().GetDim(0);
 uint32_t d = x1_shape->GetStorageShape().GetDim(1);
 uint32_t blockSize = GetOptimalBlockSize(d);
 uint32_t blockNum = (n + blockSize - 1) / blockSize;
 uint32_t totalDist = n * (n - 1) / 2;

  context->SetBlockDim(8);
  //
  tiling.SetTotalDist(totalDist);
  tiling.SetN(n);
  tiling.SetD(d);
  tiling.SetP(p);
  tiling.SetBlockSize(blockSize);
  tiling.SetBlockNum(blockNum);
  //
=======
  int32_t data_sz = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    data_sz *= x1_shape->GetStorageShape().GetDim(i);
  tiling.set_size(data_sz);
  context->SetBlockDim(8);
>>>>>>> 41a96658ee2cc44167b5b073209842e137ceacde
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class PdistGrad : public OpDef {
public:
    explicit PdistGrad(const char* name) : OpDef(name)
    {
        this->Input("grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("pdist_ouput")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
<<<<<<< HEAD
        this->AICore().AddConfig("ascend910b");
=======
        this->AICore().AddConfig("ascend910");
>>>>>>> 41a96658ee2cc44167b5b073209842e137ceacde

    }
};

OP_ADD(PdistGrad);
}
