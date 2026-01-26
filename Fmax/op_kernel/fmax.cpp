// #include "kernel_operator.h"

// extern "C" __global__ __aicore__ void fmax(GM_ADDR input, GM_ADDR other, GM_ADDR result, GM_ADDR workspace, GM_ADDR tiling) {
//     GET_TILING_DATA(tiling_data, tiling);
//     // TODO: user kernel impl
// }




#include "kernel_operator.h"

using namespace AscendC;


#define MIN(x, y) ((x) < (y) ? (x) : (y))



template <typename T>
class KernelFmax  {
public:
    __aicore__ inline KernelFmax() {}
    __aicore__ void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                         uint32_t broadcast,
                         uint32_t broadcast_mask_x1[5], uint32_t broadcast_mask_x2[5], uint32_t y_shape[5],
                         uint32_t Tile_Size,
                         TPipe* pipe) {
        broadcast_ = broadcast;
        for (uint32_t i = 0; i < 5; i++) {
            broadcast_mask_x1_[i] = broadcast_mask_x1[i];
            broadcast_mask_x2_[i] = broadcast_mask_x2[i];
            y_shape_[i] = y_shape[i];
        }
        for (uint32_t i = 0; i < 5; i++) {
            x1_shape_[i] =(broadcast_mask_x1_[i]==1) ? y_shape_[i] : 1;
            x2_shape_[i] =(broadcast_mask_x2_[i]==1) ? y_shape_[i] : 1;
        }
        sizeX1 = 1;
        sizeX2 = 1;
        sizeY = 1;
        for (uint32_t i = 0; i < 5; i++) {
            sizeX1 *= x1_shape_[i];
            sizeX2 *= x2_shape_[i];
            sizeY *= y_shape_[i];
        }

        TILE_SIZE = Tile_Size;
        x1Gm.SetGlobalBuffer((__gm__ T *)x1, sizeX1 * sizeof(T));
        x2Gm.SetGlobalBuffer((__gm__ T *)x2, sizeX2 * sizeof(T));
        yGm.SetGlobalBuffer((__gm__ T *)y, sizeY * sizeof(T));
        pipe->InitBuffer(inX1, 1, TILE_SIZE * sizeof(T));
        pipe->InitBuffer(inX2, 1, TILE_SIZE * sizeof(T));
        pipe->InitBuffer(outY, 1, TILE_SIZE * sizeof(T));
        pipe->InitBuffer(temp1Buf,  TILE_SIZE * sizeof(T));
        pipe->InitBuffer(temp2Buf,  TILE_SIZE * sizeof(T));     
        // pipe->InitBuffer(nanMask1Buf, TILE_SIZE * sizeof(uint8_t));
        // pipe->InitBuffer(nanMask2Buf, TILE_SIZE * sizeof(uint8_t));
        // pipe->InitBuffer(nanTempBuf1, TILE_SIZE * sizeof(T));
        // pipe->InitBuffer(nanTempBuf2, TILE_SIZE * sizeof(T));
        // pipe->InitBuffer(selectTempBuf, TILE_SIZE * sizeof(T));
    }

    __aicore__ void Process() {
        if (broadcast_ == 0) {
            // 非广播情况
            FmaxNonBroadcast();
        } else {
            // 广播情况
            FmaxBroadcast();
        }
    }


    __aicore__ void FmaxNonBroadcast() {
        // 非广播情况
        // TODO: user kernel impl
        for (int32_t i = GetBlockIdx() * TILE_SIZE;i < sizeX1;i+=TILE_SIZE * GetBlockNum()) {
            int32_t len = MIN(sizeX1 - i, TILE_SIZE);
            if (len<=0) break;
            CopyInX1(i, len);
            CopyInX2(i, len);
            Compute(len);
            CopyOut(i, len);  
        }
    }

    __aicore__ inline void CopyInX1(int32_t offset, int len) {
        LocalTensor<T> x1 = inX1.AllocTensor<T>();
        DataCopyExtParams copyParamsX;
        copyParamsX.blockCount = 1;
        copyParamsX.blockLen = len * sizeof(T);
        copyParamsX.srcStride = 0;
        copyParamsX.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(x1, x1Gm[offset], copyParamsX, padParams);
        inX1.EnQue(x1);
    }

    __aicore__ inline void CopyInX2(int32_t offset, int len) {
        LocalTensor<T> x2 = inX2.AllocTensor<T>();
        DataCopyExtParams copyParamsX;
        copyParamsX.blockCount = 1;
        copyParamsX.blockLen = len * sizeof(T);
        copyParamsX.srcStride = 0;
        copyParamsX.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(x2, x2Gm[offset], copyParamsX, padParams);
        inX2.EnQue(x2);
    }

    __aicore__ inline void CopyOut(int32_t offset, int len) {
        LocalTensor<T> y = outY.DeQue<T>();
        DataCopyExtParams copyParamsX;
        copyParamsX.blockCount = 1;
        copyParamsX.blockLen = len * sizeof(T);
        copyParamsX.srcStride = 0;
        copyParamsX.dstStride = 0;
        DataCopyPad(yGm[offset], y, copyParamsX);
        outY.FreeTensor(y);
    }

    __aicore__ inline void Compute(int32_t len) {
        auto x1 = inX1.DeQue<T>();
        auto x2 = inX2.DeQue<T>();
        auto y = outY.AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value) {
            for (int32_t i = 0; i < len; i++) {
                float x1_val = x1.GetValue(i);
                float x2_val = x2.GetValue(i);
                bool x1_nan = x1_val != x1_val;
                bool x2_nan = x2_val != x2_val;
                float y_val;
                if (x1_nan && x2_nan) {
                    y_val = x1_val;
                } else if (x1_nan) {
                    y_val = x2_val;
                } else if (x2_nan) {
                    y_val = x1_val;
                } else {
                    y_val = (x1_val > x2_val) ? x1_val : x2_val;
                }
                y.SetValue(i, y_val);
            }
        } else if constexpr (std::is_same<T, half>::value) {
            auto x1_f = temp1Buf.Get<float>();
            auto x2_f = temp2Buf.Get<float>();
            Cast(x1_f, x1, AscendC::RoundMode::CAST_NONE, len);
            Cast(x2_f, x2, AscendC::RoundMode::CAST_NONE, len);
            Max(y, x1, x2, len);
            for (int32_t i = 0; i < len; i++) {
                float x1_val = x1_f.GetValue(i);
                float x2_val = x2_f.GetValue(i);
                bool x1_nan = x1_val != x1_val;
                bool x2_nan = x2_val != x2_val;
                if (x1_nan && x2_nan) {
                    y.SetValue(i, x1.GetValue(i));
                } else if (x1_nan) {
                    y.SetValue(i, x2.GetValue(i));
                } else if (x2_nan) {
                    y.SetValue(i, x1.GetValue(i));
                }
            }
        } else {
            Max(y, x1, x2, len);
        }
        outY.EnQue(y);
        inX1.FreeTensor(x1);
        inX2.FreeTensor(x2);
    }



    __aicore__ void FmaxBroadcast() {
        strideX1[4] = 1;
        strideX2[4] = 1;
        strideY[4] = 1;
        for(uint32_t i = 0;i < 4;i++){
            strideX1[i] = x1_shape_[i+1] * strideX1[i+1];
            strideX2[i] = x2_shape_[i+1] * strideX2[i+1];
            strideY[i] = y_shape_[i+1] * strideY[i+1];
        }
        for(uint32_t i = 0;i < 5;i++){
            if(broadcast_mask_x1_[i]==0){
                strideX1[i] = 0;
            }
            if(broadcast_mask_x2_[i]==0){
                strideX2[i] = 0;
            }
        }
        int32_t blockIdx = GetBlockIdx();
        int32_t blockNum = GetBlockNum();
        for (int32_t i0 = 0;i0 < y_shape_[0];i0++) {
            for (int32_t i1 = 0;i1 < y_shape_[1];i1++) {
                for (int32_t i2 = 0;i2 < y_shape_[2];i2++) {
                    for (int32_t i3 = 0;i3 < y_shape_[3];i3++) {
                        for (int32_t i4 = 0;i4 < y_shape_[4];i4++) {
                            int32_t y_index = i0 * strideY[0] + i1 * strideY[1] + i2 * strideY[2] + i3 * strideY[3] + i4 * strideY[4];
                            if (y_index % blockNum != blockIdx) {
                                continue;
                            }
                            int32_t x1_index =i0 * strideX1[0] + i1 * strideX1[1] + i2 * strideX1[2] + i3 * strideX1[3] + i4 * strideX1[4];
                            int32_t x2_index =i0 * strideX2[0] + i1 * strideX2[1] + i2 * strideX2[2] + i3 * strideX2[3] + i4 * strideX2[4];
                            CopyInX1(x1_index, 1);
                            CopyInX2(x2_index, 1);
                            Compute(1);
                            CopyOut(y_index, 1);  
                        }
                    }
                }
            }
        }
    }

    uint32_t strideX1[5] = {1,1,1,1,1};
    uint32_t strideX2[5] = {1,1,1,1,1};
    uint32_t strideY[5] = {1,1,1,1,1};
    TQue<QuePosition::VECIN, 1> inX1, inX2;
    TQue<QuePosition::VECOUT, 1> outY;
    TBuf<TPosition::VECCALC> temp1Buf, temp2Buf;
    GlobalTensor<T> x1Gm, x2Gm, yGm;
    uint32_t broadcast_;
    uint32_t sizeX1, sizeX2, sizeY;
    uint32_t broadcast_mask_x1_[5];
    uint32_t broadcast_mask_x2_[5];
    uint32_t x1_shape_[5];
    uint32_t x2_shape_[5];
    uint32_t y_shape_[5];
    uint32_t TILE_SIZE;
    
};


class KernelFmax1  {
public:
    __aicore__ inline KernelFmax1() {}
    __aicore__ void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                         uint32_t broadcast,
                         uint32_t broadcast_mask_x1[5], uint32_t broadcast_mask_x2[5], uint32_t y_shape[5],
                         uint32_t Tile_Size,
                         TPipe* pipe) {
        broadcast_ = broadcast;
        for (uint32_t i = 0; i < 5; i++) {
            broadcast_mask_x1_[i] = broadcast_mask_x1[i];
            broadcast_mask_x2_[i] = broadcast_mask_x2[i];
            y_shape_[i] = y_shape[i];
        }
        for (uint32_t i = 0; i < 5; i++) {
            x1_shape_[i] =(broadcast_mask_x1_[i]==1) ? y_shape_[i] : 1;
            x2_shape_[i] =(broadcast_mask_x2_[i]==1) ? y_shape_[i] : 1;
        }
        sizeX1 = 1;
        sizeX2 = 1;
        sizeY = 1;
        for (uint32_t i = 0; i < 5; i++) {
            sizeX1 *= x1_shape_[i];
            sizeX2 *= x2_shape_[i];
            sizeY *= y_shape_[i];
        }
        TILE_SIZE = Tile_Size;
        x1Gm.SetGlobalBuffer((__gm__ int8_t *)x1, sizeX1 * sizeof(int8_t));
        x2Gm.SetGlobalBuffer((__gm__ int8_t *)x2, sizeX2 * sizeof(int8_t));
        yGm.SetGlobalBuffer((__gm__ int8_t *)y, sizeY * sizeof(int8_t));
        
        pipe->InitBuffer(temp1Buf,  TILE_SIZE * sizeof(half));
        pipe->InitBuffer(temp2Buf,  TILE_SIZE * sizeof(half));
        pipe->InitBuffer(tempYBuf,  TILE_SIZE * sizeof(half));
        
        pipe->InitBuffer(inX1, 1, TILE_SIZE * sizeof(int8_t));
        pipe->InitBuffer(inX2, 1, TILE_SIZE * sizeof(int8_t));
        pipe->InitBuffer(outY, 1, TILE_SIZE * sizeof(int8_t));   
        
    }

    __aicore__ void Process() {
        if (broadcast_ == 0) {
            // 非广播情况
            FmaxNonBroadcast();
        } else {
            // 广播情况
            FmaxBroadcast();
        }
    }


    __aicore__ void FmaxNonBroadcast() {
        // 非广播情况
        // TODO: user kernel impl
        for (int32_t i = GetBlockIdx() * TILE_SIZE;i < sizeX1;i+=TILE_SIZE * GetBlockNum()) {
            CopyInX1(i, MIN(sizeX1 - i, TILE_SIZE));
            CopyInX2(i, MIN(sizeX1 - i, TILE_SIZE));
            Compute();
            CopyOut(i, MIN(sizeX1 - i, TILE_SIZE));  
        }
    }

    __aicore__ inline void CopyInX1(int32_t offset, int len) {
        LocalTensor<int8_t> x1 = inX1.AllocTensor<int8_t>();
        DataCopyExtParams copyParamsX;
        copyParamsX.blockCount = 1;
        copyParamsX.blockLen = len * sizeof(int8_t);
        copyParamsX.srcStride = 0;
        copyParamsX.dstStride = 0;
        DataCopyPadExtParams<int8_t> padParams{false, 0, 0, 0};
        DataCopyPad(x1, x1Gm[offset], copyParamsX, padParams);
        inX1.EnQue(x1);
    }

    __aicore__ inline void CopyInX2(int32_t offset, int len) {
        
        LocalTensor<int8_t> x2 = inX2.AllocTensor<int8_t>();
        DataCopyExtParams copyParamsX;
        copyParamsX.blockCount = 1;
        copyParamsX.blockLen = len * sizeof(int8_t);
        copyParamsX.srcStride = 0;
        copyParamsX.dstStride = 0;
        DataCopyPadExtParams<int8_t> padParams{false, 0, 0, 0};
        DataCopyPad(x2, x2Gm[offset], copyParamsX, padParams);
        inX2.EnQue(x2);
    }

    __aicore__ inline void CopyOut(int32_t offset, int len) {
        LocalTensor<int8_t> y = outY.DeQue<int8_t>();
        DataCopyExtParams copyParamsX;
        copyParamsX.blockCount = 1;
        copyParamsX.blockLen = len * sizeof(int8_t);
        copyParamsX.srcStride = 0;
        copyParamsX.dstStride = 0;
        DataCopyPad(yGm[offset], y, copyParamsX);
        outY.FreeTensor(y);
    }

    __aicore__ inline void Compute() {
        if constexpr (std::is_same<DTYPE_Y, uint8_t>::value) {
            auto x1 = inX1.DeQue<uint8_t>();
            auto x2 = inX2.DeQue<uint8_t>();
            auto y = outY.AllocTensor<uint8_t>();
                    auto half_x1 =temp1Buf.Get<half>();
            auto half_x2 =temp2Buf.Get<half>();
            auto half_y = tempYBuf.Get<half>();
            Cast(half_x1,x1,AscendC::RoundMode::CAST_NONE,TILE_SIZE);
            Cast(half_x2,x2,AscendC::RoundMode::CAST_NONE,TILE_SIZE);
            Max(half_y,half_x1,half_x2,TILE_SIZE); 
            Cast(y,half_y,AscendC::RoundMode::CAST_NONE,TILE_SIZE);
            outY.EnQue(y);
            inX1.FreeTensor(x1);
            inX2.FreeTensor(x2);
        }
        else  {
            auto x1 = inX1.DeQue<int8_t>();
            auto x2 = inX2.DeQue<int8_t>();
            auto y = outY.AllocTensor<int8_t>();
                    auto half_x1 =temp1Buf.Get<half>();
            auto half_x2 =temp2Buf.Get<half>();
            auto half_y = tempYBuf.Get<half>();
            Cast(half_x1,x1,AscendC::RoundMode::CAST_NONE,TILE_SIZE);
            Cast(half_x2,x2,AscendC::RoundMode::CAST_NONE,TILE_SIZE);
            Max(half_y,half_x1,half_x2,TILE_SIZE); 
            Cast(y,half_y,AscendC::RoundMode::CAST_NONE,TILE_SIZE);
            outY.EnQue(y);
            inX1.FreeTensor(x1);
            inX2.FreeTensor(x2);
        }

    }



    __aicore__ void FmaxBroadcast() {
        strideX1[4] = 1;
        strideX2[4] = 1;
        strideY[4] = 1;
        for(uint32_t i = 0;i < 4;i++){
            strideX1[i] = x1_shape_[i+1] * strideX1[i+1];
            strideX2[i] = x2_shape_[i+1] * strideX2[i+1];
            strideY[i] = y_shape_[i+1] * strideY[i+1];
        }
        for(uint32_t i = 0;i < 5;i++){
            if(broadcast_mask_x1_[i]==0){
                strideX1[i] = 0;
            }
            if(broadcast_mask_x2_[i]==0){
                strideX2[i] = 0;
            }
        }
        int32_t blockIdx = GetBlockIdx();
        int32_t blockNum = GetBlockNum();
        for (int32_t i0 = 0;i0 < y_shape_[0];i0++) {
            for (int32_t i1 = 0;i1 < y_shape_[1];i1++) {
                for (int32_t i2 = 0;i2 < y_shape_[2];i2++) {
                    for (int32_t i3 = 0;i3 < y_shape_[3];i3++) {
                        for (int32_t i4 = 0;i4 < y_shape_[4];i4++) {
                            int32_t y_index = i0 * strideY[0] + i1 * strideY[1] + i2 * strideY[2] + i3 * strideY[3] + i4 * strideY[4];
                            if (y_index % blockNum != blockIdx) {
                                continue;
                            }
                            int32_t x1_index =i0 * strideX1[0] + i1 * strideX1[1] + i2 * strideX1[2] + i3 * strideX1[3] + i4 * strideX1[4];
                            int32_t x2_index =i0 * strideX2[0] + i1 * strideX2[1] + i2 * strideX2[2] + i3 * strideX2[3] + i4 * strideX2[4];
                            if constexpr (std::is_same<DTYPE_Y, uint8_t>::value) {
                                uint8_t x1_val = x1Gm.GetValue(x1_index);
                                uint8_t x2_val = x2Gm.GetValue(x2_index);
                                uint8_t y_val = (x1_val > x2_val) ? x1_val : x2_val;
                                yGm.SetValue(y_index, y_val);
                            }
                            else  {
                                int8_t x1_val = x1Gm.GetValue(x1_index);
                                int8_t x2_val = x2Gm.GetValue(x2_index);
                                int8_t y_val = (x1_val > x2_val) ? x1_val : x2_val;
                                yGm.SetValue(y_index, y_val);
                            }
                        }
                    }
                }
            }
        }
    }

    uint32_t strideX1[5] = {1,1,1,1,1};
    uint32_t strideX2[5] = {1,1,1,1,1};
    uint32_t strideY[5] = {1,1,1,1,1};

    TQue<QuePosition::VECIN, 1> inX1, inX2;
    TQue<QuePosition::VECOUT, 1> outY;
    TBuf<TPosition::VECCALC> temp2Buf,temp1Buf,tempYBuf;
    GlobalTensor<int8_t> x1Gm, x2Gm, yGm;
    uint32_t broadcast_;
    uint32_t sizeX1, sizeX2, sizeY;
    uint32_t broadcast_mask_x1_[5];
    uint32_t broadcast_mask_x2_[5];
    uint32_t x1_shape_[5];
    uint32_t x2_shape_[5];
    uint32_t y_shape_[5];
    uint32_t TILE_SIZE;
    
};


class KernelFmax2  {
public:
    __aicore__ inline KernelFmax2() {}
    __aicore__ void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                         uint32_t broadcast,
                         uint32_t broadcast_mask_x1[5], uint32_t broadcast_mask_x2[5], uint32_t y_shape[5],
                         uint32_t Tile_Size,
                         TPipe* pipe) {
        broadcast_ = broadcast;
        for (uint32_t i = 0; i < 5; i++) {
            broadcast_mask_x1_[i] = broadcast_mask_x1[i];
            broadcast_mask_x2_[i] = broadcast_mask_x2[i];
            y_shape_[i] = y_shape[i];
        }
        for (uint32_t i = 0; i < 5; i++) {
            x1_shape_[i] =(broadcast_mask_x1_[i]==1) ? y_shape_[i] : 1;
            x2_shape_[i] =(broadcast_mask_x2_[i]==1) ? y_shape_[i] : 1;
        }
        sizeX1 = 1;
        sizeX2 = 1;
        sizeY = 1;
        for (uint32_t i = 0; i < 5; i++) {
            sizeX1 *= x1_shape_[i];
            sizeX2 *= x2_shape_[i];
            sizeY *= y_shape_[i];
        }
        TILE_SIZE = Tile_Size;
        x1Gm.SetGlobalBuffer((__gm__ int32_t *)x1, sizeX1 * sizeof(int64_t));
        x2Gm.SetGlobalBuffer((__gm__ int32_t *)x2, sizeX2 * sizeof(int64_t));
        yGm.SetGlobalBuffer((__gm__ int32_t *)y, sizeY * sizeof(int64_t));
    }

    __aicore__ void Process() {
        strideX1[4] = 1;
        strideX2[4] = 1;
        strideY[4] = 1;
        for(uint32_t i = 0;i < 4;i++){
            strideX1[i] = x1_shape_[i+1] * strideX1[i+1];
            strideX2[i] = x2_shape_[i+1] * strideX2[i+1];
            strideY[i] = y_shape_[i+1] * strideY[i+1];
        }
        for(uint32_t i = 0;i < 5;i++){
            if(broadcast_mask_x1_[i]==0){
                strideX1[i] = 0;
            }
            if(broadcast_mask_x2_[i]==0){
                strideX2[i] = 0;
            }
        }
        int32_t blockIdx = GetBlockIdx();
        int32_t blockNum = GetBlockNum();
        for (int32_t i0 = 0;i0 < y_shape_[0];i0++) {
            for (int32_t i1 = 0;i1 < y_shape_[1];i1++) {
                for (int32_t i2 = 0;i2 < y_shape_[2];i2++) {
                    for (int32_t i3 = 0;i3 < y_shape_[3];i3++) {
                        for (int32_t i4 = 0;i4 < y_shape_[4];i4++) {
                            int32_t y_index = i0 * strideY[0] + i1 * strideY[1] + i2 * strideY[2] + i3 * strideY[3] + i4 * strideY[4];
                            if (y_index % blockNum != blockIdx) {
                                continue;
                            }
                            int32_t x1_index =i0 * strideX1[0] + i1 * strideX1[1] + i2 * strideX1[2] + i3 * strideX1[3] + i4 * strideX1[4];
                            int32_t x2_index =i0 * strideX2[0] + i1 * strideX2[1] + i2 * strideX2[2] + i3 * strideX2[3] + i4 * strideX2[4];
                                                    /********** int64_t拆分处理 **********/
                            // 1. 拆分64位为两个32位（高位+低位），通过uint32_t接口访问GM
                            uint32_t x1_lo = x1Gm.GetValue(x1_index * 2);
                            uint32_t x1_hi = x1Gm.GetValue(x1_index * 2 + 1);
                            uint32_t x2_lo = x2Gm.GetValue(x2_index * 2);
                            uint32_t x2_hi = x2Gm.GetValue(x2_index * 2 + 1);
                            // 2. 64位比较逻辑：先比高位，高位相等再比低位
                            uint32_t y_hi, y_lo;
                            // 有符号int64_t比较：需转换为int32_t判断高位
                            int32_t x1_hi_s = static_cast<int32_t>(x1_hi);
                            int32_t x2_hi_s = static_cast<int32_t>(x2_hi);
                            if (x1_hi_s > x2_hi_s) {
                                y_hi = x1_hi;
                                y_lo = x1_lo;
                            } else if (x1_hi_s < x2_hi_s) {
                                y_hi = x2_hi;
                                y_lo = x2_lo;
                            } else {
                                // 高位相等，比较低位
                                y_hi = x1_hi;
                                y_lo = (x1_lo > x2_lo) ? x1_lo : x2_lo;
                            }
                            // 3. 将32位结果写回GM（合并为64位）
                            yGm.SetValue(y_index * 2, y_lo);
                            yGm.SetValue(y_index * 2 + 1, y_hi);
                        }
                    }
                }
            }
        }
    }

    uint32_t strideX1[5] = {1,1,1,1,1};
    uint32_t strideX2[5] = {1,1,1,1,1};
    uint32_t strideY[5] = {1,1,1,1,1};
    GlobalTensor<int32_t> x1Gm, x2Gm, yGm;
    uint32_t broadcast_;
    uint32_t sizeX1, sizeX2, sizeY;
    uint32_t broadcast_mask_x1_[5];
    uint32_t broadcast_mask_x2_[5];
    uint32_t x1_shape_[5];
    uint32_t x2_shape_[5];
    uint32_t y_shape_[5];
    uint32_t TILE_SIZE;
    
};




extern "C" __global__ __aicore__ void fmax(GM_ADDR input, GM_ADDR other, GM_ADDR result, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    TPipe pipe;
    if constexpr (std::is_same<DTYPE_Y, int16_t>::value || std::is_same<DTYPE_Y, float>::value || std::is_same<DTYPE_Y,
                    half>::value || std::is_same<DTYPE_Y, int32_t>::value) {
        KernelFmax<DTYPE_Y> op;
        op.Init(input, other, result, 
                tiling_data.broadcast,
                tiling_data.broadcast_mask_x1, tiling_data.broadcast_mask_x2, tiling_data.y_shape,
                tiling_data.Tile_Size,
                &pipe);
        op.Process(); 
    }
    else if (std::is_same<DTYPE_Y, int8_t>::value || std::is_same<DTYPE_Y, bool>::value||std::is_same<DTYPE_Y, uint8_t>::value) {
        KernelFmax1 op;
        op.Init(input, other, result, 
                tiling_data.broadcast,
                tiling_data.broadcast_mask_x1, tiling_data.broadcast_mask_x2, tiling_data.y_shape,
                tiling_data.Tile_Size,
                &pipe);
        op.Process(); 
    }
    else if (std::is_same<DTYPE_Y, int64_t>::value) {
        KernelFmax2 op;
        op.Init(input, other, result, 
                tiling_data.broadcast,
                tiling_data.broadcast_mask_x1, tiling_data.broadcast_mask_x2, tiling_data.y_shape,
                tiling_data.Tile_Size,
                &pipe);
        op.Process();
    }

}
