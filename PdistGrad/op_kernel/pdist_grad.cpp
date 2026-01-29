#include "kernel_operator.h"

using namespace AscendC;

class PdistGradKernel {
public:
    __aicore__ inline PdistGradKernel() {}
    
    __aicore__ inline void Init(GM_ADDR grad,           // grad_dist的梯度 [totalDist]
                                GM_ADDR input,          // 输入x [n, d]
                                GM_ADDR pdist_output,   // 前向pdist的输出（可选，用于距离缓存）
                                GM_ADDR result,         // 输出的梯度 [n, d]
                                uint32_t n, uint32_t d, float p, 
                                uint32_t totalDist,  uint32_t blockNum,
                                uint32_t blockSize) {
        this->n = n;
        this->d = d;
        this->p = p;
        this->totalDist = totalDist;
        this->blockNum = blockNum;
        this->blockId = GetBlockIdx();
        
        //  关键：设置分块参数
        this->blockSize = blockSize;  // 自动选择分块大小
        this->numBlocks = (d + blockSize - 1) / blockSize;  // 计算分块数量
        
        this->gradGm.SetGlobalBuffer((__gm__ float*)grad, totalDist);
        this->inputGm.SetGlobalBuffer((__gm__ float*)input, n * d);
        this->pdistOutputGm.SetGlobalBuffer((__gm__ float*)pdist_output, totalDist);
        this->resultGm.SetGlobalBuffer((__gm__ float*)result, n * d);
        
        //  修改：Pipe缓冲区按分块大小设计
        // 每个buffer只存一个分块，而不是整个d维度
        const uint32_t BLOCK_BYTES = blockSize * sizeof(float);
        pipe.InitBuffer(inQueueInput1, BUFFER_NUM, BLOCK_BYTES * 2);   // 两个向量的一个分块
        pipe.InitBuffer(inQueueInput2, BUFFER_NUM, BLOCK_BYTES * 2);   // 两个向量的一个分块
        pipe.InitBuffer(tempBuf, BLOCK_BYTES * 4);      // 中间结果
        pipe.InitBuffer(halfBuf, 3*blockSize*sizeof(half));      // 中间结果
        pipe.InitBuffer(gradBuf, BLOCK_BYTES );      // 中间结果
        pipe.InitBuffer(powerBuf, 3*sizeof(float));      // 中间结果
        pipe.InitBuffer(cmpBuf, 2*blockSize*sizeof(uint8_t));      // 中间结果
        pipe.InitBuffer(syncQueue, BUFFER_NUM, sizeof(uint32_t));     // 同步
        
        // 初始化result为0
        InitResultToZero();
    }
    

    
    // 初始化result为0（已优化，使用分块）
    __aicore__ inline void InitResultToZero() {
        uint32_t elementsPerCore = (n * d + blockNum - 1) / blockNum;
        uint32_t start = elementsPerCore * blockId;
        uint32_t end = min(start + elementsPerCore, n * d);
        //  分块初始化
        for (uint32_t elem = start; elem < end; elem += blockSize) {
            uint32_t remaining = end - elem;
            uint32_t currentBlockSize = min(blockSize, remaining);
            // 设置当前分块为0
            for (uint32_t i = 0; i < currentBlockSize; ++i) {
                resultGm.SetValue(elem + i, 0.0f);
            }
        }
    }
    
    __aicore__ inline void Process() {
        uint32_t distPerCore = totalDist / blockNum;
        uint32_t distStart = distPerCore * blockId;
        uint32_t distEnd = (blockId == blockNum - 1) ? totalDist : distStart + distPerCore;
        for (uint32_t idx = distStart; idx < distEnd; ++idx) {
            ProcessOnePair(idx);
        }
        ProcessComplete();
    }
    
private:
    // 分块处理一个(i,j)对
    // 加载梯度值
    // 计算或获取距离
    //  分块计算并累加梯度
    __aicore__ inline void ProcessOnePair(uint32_t distIdx) {
        uint32_t i, j;
        DistIdxToPair(distIdx, i, j);
        float gradValue = gradGm.GetValue(distIdx);
        if (abs(gradValue) < 1e-12f) return;
        float distance = pdistOutputGm.GetValue(distIdx);
        if (distance < 1e-8f) return;
        ComputeGradientByBlocks(i, j, gradValue, distance);
    }
    
    //  新增：分块计算并累加梯度
    __aicore__ inline void Getscale(float distance, float gradValue, float& scale){
        if (p == 2.0f) {
                    scale = gradValue / distance;
                } 
        else if (p == 1.0f) {
                    scale = gradValue;
                }  
        else if (!IsInfinity(p)) {
                    LocalTensor<float> baseTensor = powerBuf.GetWithOffset<float>(0, 1);
                    LocalTensor<float> powerResult = powerBuf.GetWithOffset<float>(1,1);
                    baseTensor.SetValue(0, distance);
                    float exp=p-1.0f;
                    // Power(dstTensor, srcTensor1, scalarValue)
                    Power<float, false>(powerResult,baseTensor,exp);
                    // 计算 scale = gradValue / distance^(p-1)
                    float denominator = powerResult.GetValue(0);
                    if (denominator > 1e-12f) {
                        scale = gradValue / denominator;
                    } else {
                        scale = gradValue * 1e8f; // 防止除0
                    }
                }
    }
    __aicore__ inline void ComputeGradientByBlocks(uint32_t i, uint32_t j, 
                                                   float gradValue, float distance) {
        // 预计算缩放因子=gradValue/distance^(p-1)
        float scale = 1.0f;
        Getscale(distance, gradValue, scale);
        
        // 对于L∞距离，需要先找到全局最大值位置
        uint32_t maxIdxGlobal = 0;
        // if (IsInfinity(p)) {
        //     maxIdxGlobal = FindGlobalMaxIndex(i, j);
        // }
        
        // 遍历所有分块计算梯度
        for (uint32_t block = 0; block < numBlocks; ++block) {
            uint32_t blockStart = block * blockSize;
            uint32_t currentBlockSize = (block == numBlocks - 1) ? (d - blockStart) : blockSize;
            // Copy IN
            LocalTensor<float> vec_i_block = inQueueInput1.AllocTensor<float>();
            LocalTensor<float> vec_j_block = inQueueInput2.AllocTensor<float>();
            
            DataCopy(vec_i_block, inputGm[i * d + blockStart], currentBlockSize);
            DataCopy(vec_j_block, inputGm[j * d + blockStart], currentBlockSize);
            // Compute

            LocalTensor<float> gradBlock = gradBuf.GetWithOffset<float>(0, currentBlockSize);
            ComputeGradientBlock(vec_i_block, vec_j_block, scale, gradBlock, currentBlockSize);

            // 累加梯度到result 
            AccumulateGradientBlockI(resultGm, i * d + blockStart, gradBlock, currentBlockSize);
            AccumulateGradientBlockJ(resultGm, j * d + blockStart, gradBlock, currentBlockSize);
            
            inQueueInput1.FreeTensor(vec_i_block);
            inQueueInput2.FreeTensor(vec_j_block);
        }
    }
    
    
    //  新增：计算分块梯度（通用p范数）
    __aicore__ inline void ComputeGradientBlock(LocalTensor<float>& a, LocalTensor<float>& b,
                                               float scale, LocalTensor<float>& grad,
                                               uint32_t len) {
        LocalTensor<float> diff = tempBuf.GetWithOffset<float>(0, len);
        Sub(diff, a, b,len); 
        if (p == 2.0f) {
            Muls(grad, diff, scale,len);
        } else {
            // sign = diff > 0 ? 1.0f : (diff < 0 ? -1.0f : 0.0f)
            LocalTensor<uint8_t> GT0Result = cmpBuf.GetWithOffset<uint8_t>(0, len);
            LocalTensor<uint8_t> LT0Result= cmpBuf.GetWithOffset<uint8_t>(len, len);
            //
            LocalTensor<half> halfTemp1 = halfBuf.GetWithOffset<half>(0, len);
            LocalTensor<half> halfTemp2 = halfBuf.GetWithOffset<half>(len, len);
            LocalTensor<half> halfTemp3 = halfBuf.GetWithOffset<half>(len*2, len);
            //
            LocalTensor<float>  signResult= tempBuf.GetWithOffset<float>(len, len);
            CompareScalar(GT0Result, diff, 0.0f, CMPMODE::GT,len);
            CompareScalar(LT0Result, diff, 0.0f, CMPMODE::LT,len);
            Cast<half>( halfTemp1, GT0Result,AscendC::RoundMode::CAST_NONE,len);
            Cast<half>( halfTemp2, LT0Result,AscendC::RoundMode::CAST_NONE,len);
            Sub(halfTemp3, halfTemp1, halfTemp2,len);//if GT0Result=1 1 or LT0Result=1 -1else 0
            Cast<float>( signResult,halfTemp3,AscendC::RoundMode::CAST_NONE,len);
            //
            if (p == 1.0f) {
                Muls(grad, signResult, scale,len);
            }
            else {
                // 通用p范数梯度
                LocalTensor<float> absDiff = tempBuf.GetWithOffset<float>(len*2, len);
                Abs(absDiff, diff, len);
                LocalTensor<float> power2Result = tempBuf.GetWithOffset<float>(len*3, len);
                Power<float, false>(power2Result,  absDiff,float(p-1.0f));
                Mul(grad, power2Result, signResult,len);
                Muls(grad, grad, scale,len);
                // for (uint32_t k = 0; k < len; ++k) {
                //     float sign = diff.GetValue(k) > 0 ? 1.0f : (diff.GetValue(k) < 0 ? -1.0f : 0.0f);    
                //     if (absDiff > 1e-12f) {
                //         grad.SetValue(k, sign * pow(absDiff, p - 1.0f) * scale);
                //     } else {
                //         grad.SetValue(k, 0.0f);
                //     }
                // }
            }
        }
        
        
        
    }
    

    
    __aicore__ inline void AccumulateGradientBlockI(GlobalTensor<float>& result_gm,
                                              uint32_t offset,
                                              LocalTensor<float>& grad_contrib,
                                              uint32_t len) {        
        SetAtomicNone();
        SetAtomicAdd<float>();
        DataCopy(result_gm[offset], grad_contrib, len);
        SetAtomicNone();
    }
    __aicore__ inline void AccumulateGradientBlockJ(GlobalTensor<float>& result_gm,
                                              uint32_t offset,
                                              LocalTensor<float>& grad_contrib,
                                              uint32_t len) {        
        Muls(grad_contrib, grad_contrib, float(-1.0f), len);
        SetAtomicNone();
        SetAtomicAdd<float>();
        DataCopy(result_gm[offset], grad_contrib, len);
        SetAtomicNone();
    }
    // 其他原有函数保持不变...
    __aicore__ inline void DistIdxToPair(uint32_t idx, uint32_t& i, uint32_t& j) {
        i = 0;
        while (idx >= n - i - 1) {
            idx -= n - i - 1;
            i++;
        }
        j = i + idx + 1;
    }
    
    __aicore__ inline bool IsInfinity(float x) {
        const float INF_THRESHOLD = 1e10f;
        return x >= INF_THRESHOLD;
    }
    
    __aicore__ inline void ProcessComplete() {
        LocalTensor<uint32_t> syncFlag = syncQueue.AllocTensor<uint32_t>();
        syncFlag.SetValue(0, blockId);
        syncQueue.EnQue(syncFlag);
        syncQueue.FreeTensor(syncFlag);
    }
    
private:
    uint32_t n;           // 向量数量
    uint32_t d;           // 向量维度
    float p;             // p值
    uint32_t blockNum;   // 任务块数
    uint32_t blockId;    // 当前块id
    uint32_t totalDist;  // 总距离数量
    
    //  新增：分块处理参数
    uint32_t blockSize;  // 每个分块的大小
    uint32_t numBlocks;  // 总的分块数量
    
    GlobalTensor<float> gradGm;
    GlobalTensor<float> inputGm;
    GlobalTensor<float> pdistOutputGm;
    GlobalTensor<float> resultGm;
    
    TPipe pipe;
    static constexpr int BUFFER_NUM = 2;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInput1,inQueueInput2;
    TBuf<TPosition::VECCALC> tempBuf,gradBuf,powerBuf,cmpBuf,halfBuf;
    TQue<QuePosition::VECOUT, BUFFER_NUM> syncQueue;
    
};

extern "C" __global__ __aicore__ void pdist_grad(GM_ADDR grad, GM_ADDR input, GM_ADDR pdist_output, GM_ADDR result, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    PdistGradKernel kernel;
    kernel.Init(grad, input, pdist_output, result,
                tiling_data.n, tiling_data.d, tiling_data.p, 
                tiling_data.totalDist, tiling_data.blockNum,
                tiling_data.blockSize);
    kernel.Process();
}

    // //  新增：计算切比雪夫梯度分块
    // __aicore__ inline void ComputeChebyshevGradientBlock(LocalTensor<float>& a, LocalTensor<float>& b,
    //                                                     float gradValue, uint32_t globalMaxIdx,
    //                                                     uint32_t blockStart, LocalTensor<float>& grad,
    //                                                     uint32_t len) {
    //     // 初始化全0
    //     for (uint32_t k = 0; k < len; ++k) {
    //         grad.SetValue(k, 0.0f);
    //     }
        
    //     // 检查全局最大值是否在当前分块内
    //     if (globalMaxIdx >= blockStart && globalMaxIdx < blockStart + len) {
    //         uint32_t localIdx = globalMaxIdx - blockStart;
    //         float diff = a.GetValue(localIdx) - b.GetValue(localIdx);
    //         grad.SetValue(localIdx, (diff > 0 ? gradValue : -gradValue));
    //     }
    // }
    
    // //  新增：找到全局最大值索引（用于L∞距离）
    // __aicore__ inline uint32_t FindGlobalMaxIndex(uint32_t i, uint32_t j) {
    //     float maxVal = 0.0f;
    //     uint32_t maxIdx = 0;
        
    //     for (uint32_t block = 0; block < numBlocks; ++block) {
    //         uint32_t blockStart = block * blockSize;
    //         uint32_t currentBlockSize = (block == numBlocks - 1) ? (d - blockStart) : blockSize;
            
    //         LocalTensor<float> vec_i_block = inQueueInput.AllocTensor<float>();
    //         LocalTensor<float> vec_j_block = inQueueInput.AllocTensor<float>();
            
    //         LoadVectorBlock(inputGm, i * d + blockStart, vec_i_block, currentBlockSize);
    //         LoadVectorBlock(inputGm, j * d + blockStart, vec_j_block, currentBlockSize);
            
    //         for (uint32_t k = 0; k < currentBlockSize; ++k) {
    //             float diff = vec_i_block.GetValue(k) - vec_j_block.GetValue(k);
    //             float absDiff = abs(diff);
    //             if (absDiff > maxVal) {
    //                 maxVal = absDiff;
    //                 maxIdx = blockStart + k;
    //             }
    //         }
            
    //         inQueueInput.FreeTensor(vec_i_block);
    //         inQueueInput.FreeTensor(vec_j_block);
    //     }
        
    //     return maxIdx;
    // }