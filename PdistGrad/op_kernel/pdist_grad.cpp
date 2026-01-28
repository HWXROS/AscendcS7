#include "kernel_operator.h"

<<<<<<< HEAD
using namespace AscendC;

class PdistGradKernel {
public:
    __aicore__ inline PdistGradKernel() {}
    
    __aicore__ inline void Init(GM_ADDR grad,           // grad_dist的梯度 [totalDist]
                                GM_ADDR input,          // 输入x [n, d]
                                GM_ADDR pdist_output,   // 前向pdist的输出（可选，用于距离缓存）
                                GM_ADDR result,         // 输出的梯度 [n, d]
                                uint32_t n, uint32_t d, float p, 
                                uint32_t totalDist, uint32_t tilingKey, uint32_t blocknum,
                                uint32_t blockSize) {
        this->n = n;
        this->d = d;
        this->p = p;
        this->totalDist = totalDist;
        this->tilingKey = tilingKey;
        this->blocknum = blocknum;
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
        pipe.InitBuffer(inQueueInput, BUFFER_NUM, BLOCK_BYTES * 2);   // 两个向量的一个分块
        pipe.InitBuffer(inQueueGrad, BUFFER_NUM, sizeof(float));      // 存grad值
        pipe.InitBuffer(tempQueue, BUFFER_NUM, BLOCK_BYTES * 3);      // 中间结果
        pipe.InitBuffer(syncQueue, BUFFER_NUM, sizeof(uint32_t));     // 同步
        
        // 初始化result为0
        InitResultToZero();
    }
    

    
    // 初始化result为0（已优化，使用分块）
    __aicore__ inline void InitResultToZero() {
        uint32_t elementsPerCore = (n * d + blocknum - 1) / blocknum;
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
        uint32_t distPerCore = totalDist / blocknum;
        uint32_t distStart = distPerCore * blockId;
        uint32_t distEnd = (blockId == blocknum - 1) ? totalDist : distStart + distPerCore;
        
        for (uint32_t idx = distStart; idx < distEnd; ++idx) {
            ProcessOnePair(idx);
        }
        
        ProcessComplete();
    }
    
private:
    //  重构：分块处理一个(i,j)对
    __aicore__ inline void ProcessOnePair(uint32_t distIdx) {
        uint32_t i, j;
        DistIdxToPair(distIdx, i, j);
        
        // 加载梯度值
        float gradValue = gradGm.GetValue(distIdx);
        if (abs(gradValue) < 1e-12f) return;
        // 计算或获取距离
        float distance = pdistOutputGm.GetValue(distIdx);
        if (distance < 1e-8f) return;
        //  分块计算并累加梯度
        ComputeGradientByBlocks(i, j, gradValue, distance);
    }
    
    //  新增：分块计算并累加梯度
    __aicore__ inline void ComputeGradientByBlocks(uint32_t i, uint32_t j, 
                                                   float gradValue, float distance) {
        // 预计算缩放因子
        float scale = 1.0f;
        if (p == 2.0f) {
            scale = gradValue / distance;
        } else if (p == 1.0f) {
            scale = gradValue;
        } else if (!IsInfinity(p)) {
            scale = gradValue / pow(distance, p - 1.0f);
        }
        
        // 对于L∞距离，需要先找到全局最大值位置
        uint32_t maxIdxGlobal = 0;
        if (IsInfinity(p)) {
            maxIdxGlobal = FindGlobalMaxIndex(i, j);
        }
        
        // 遍历所有分块计算梯度
        for (uint32_t block = 0; block < numBlocks; ++block) {
            uint32_t blockStart = block * blockSize;
            uint32_t currentBlockSize = (block == numBlocks - 1) ? (d - blockStart) : blockSize;
            
            // 加载当前分块
            LocalTensor<float> vec_i_block = inQueueInput.AllocTensor<float>();
            LocalTensor<float> vec_j_block = inQueueInput.AllocTensor<float>();
            
            LoadVectorBlock(inputGm, i * d + blockStart, vec_i_block, currentBlockSize);
            LoadVectorBlock(inputGm, j * d + blockStart, vec_j_block, currentBlockSize);
            
            // 计算当前分块的梯度贡献
            LocalTensor<float> gradBlock = tempQueue.AllocTensor<float>();
            
            if (IsInfinity(p)) {
                // L∞梯度：只在全局最大值位置计算
                ComputeChebyshevGradientBlock(vec_i_block, vec_j_block, gradValue, 
                                             maxIdxGlobal, blockStart, gradBlock, currentBlockSize);
            } else {
                // 其他p范数梯度
                ComputeGradientBlock(vec_i_block, vec_j_block, scale, gradBlock, currentBlockSize);
            }
            
            // 累加梯度到result
            AccumulateGradientBlock(resultGm, i * d + blockStart, gradBlock, currentBlockSize, 1.0f);
            AccumulateGradientBlock(resultGm, j * d + blockStart, gradBlock, currentBlockSize, -1.0f);
            
            inQueueInput.FreeTensor(vec_i_block);
            inQueueInput.FreeTensor(vec_j_block);
            tempQueue.FreeTensor(gradBlock);
        }
    }
    
    //  新增：分块加载向量
    __aicore__ inline void LoadVectorBlock(GlobalTensor<float>& gm, uint32_t offset,
                                          LocalTensor<float>& local, uint32_t len) {
        for (uint32_t i = 0; i < len; ++i) {
            local.SetValue(i, gm.GetValue(offset + i));
        }
    }
    
    //  新增：计算分块梯度（通用p范数）
    __aicore__ inline void ComputeGradientBlock(LocalTensor<float>& a, LocalTensor<float>& b,
                                               float scale, LocalTensor<float>& grad,
                                               uint32_t len) {
        if (p == 2.0f) {
            // L2梯度
            for (uint32_t k = 0; k < len; ++k) {
                float diff = a.GetValue(k) - b.GetValue(k);
                grad.SetValue(k, diff * scale);
            }
        } else if (p == 1.0f) {
            // L1梯度
            for (uint32_t k = 0; k < len; ++k) {
                float diff = a.GetValue(k) - b.GetValue(k);
                grad.SetValue(k, (diff > 0 ? scale : (diff < 0 ? -scale : 0)));
            }
        } else {
            // 通用p范数梯度
            for (uint32_t k = 0; k < len; ++k) {
                float diff = a.GetValue(k) - b.GetValue(k);
                float absDiff = abs(diff);
                float sign = diff > 0 ? 1.0f : (diff < 0 ? -1.0f : 0.0f);
                
                if (absDiff > 1e-12f) {
                    grad.SetValue(k, sign * pow(absDiff, p - 1.0f) * scale);
                } else {
                    grad.SetValue(k, 0.0f);
                }
            }
        }
    }
    
    //  新增：计算切比雪夫梯度分块
    __aicore__ inline void ComputeChebyshevGradientBlock(LocalTensor<float>& a, LocalTensor<float>& b,
                                                        float gradValue, uint32_t globalMaxIdx,
                                                        uint32_t blockStart, LocalTensor<float>& grad,
                                                        uint32_t len) {
        // 初始化全0
        for (uint32_t k = 0; k < len; ++k) {
            grad.SetValue(k, 0.0f);
        }
        
        // 检查全局最大值是否在当前分块内
        if (globalMaxIdx >= blockStart && globalMaxIdx < blockStart + len) {
            uint32_t localIdx = globalMaxIdx - blockStart;
            float diff = a.GetValue(localIdx) - b.GetValue(localIdx);
            grad.SetValue(localIdx, (diff > 0 ? gradValue : -gradValue));
        }
    }
    
    //  新增：找到全局最大值索引（用于L∞距离）
    __aicore__ inline uint32_t FindGlobalMaxIndex(uint32_t i, uint32_t j) {
        float maxVal = 0.0f;
        uint32_t maxIdx = 0;
        
        for (uint32_t block = 0; block < numBlocks; ++block) {
            uint32_t blockStart = block * blockSize;
            uint32_t currentBlockSize = (block == numBlocks - 1) ? (d - blockStart) : blockSize;
            
            LocalTensor<float> vec_i_block = inQueueInput.AllocTensor<float>();
            LocalTensor<float> vec_j_block = inQueueInput.AllocTensor<float>();
            
            LoadVectorBlock(inputGm, i * d + blockStart, vec_i_block, currentBlockSize);
            LoadVectorBlock(inputGm, j * d + blockStart, vec_j_block, currentBlockSize);
            
            for (uint32_t k = 0; k < currentBlockSize; ++k) {
                float diff = vec_i_block.GetValue(k) - vec_j_block.GetValue(k);
                float absDiff = abs(diff);
                if (absDiff > maxVal) {
                    maxVal = absDiff;
                    maxIdx = blockStart + k;
                }
            }
            
            inQueueInput.FreeTensor(vec_i_block);
            inQueueInput.FreeTensor(vec_j_block);
        }
        
        return maxIdx;
    }
    
    //  优化：累加分块梯度（向量化）
    __aicore__ inline void AccumulateGradientBlock(GlobalTensor<float>& result_gm,
                                                  uint32_t offset,
                                                  LocalTensor<float>& grad_contrib,
                                                  uint32_t len, float sign) {
        constexpr uint32_t VEC_LEN = 8;
        
        for (uint32_t start = 0; start < len; start += VEC_LEN) {
            uint32_t remaining = len - start;j-
            uint32_t currentVecLen = min(VEC_LEN, remaining);
            
            for (uint32_t v = 0; v < currentVecLen; ++v) {
                float value = grad_contrib.GetValue(start + v) * sign;
                __atomic_add_f32(result_gm.GetAddr(offset + start + v), value);
            }
        }
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
    uint32_t blocknum;   // 任务块数
    uint32_t blockId;    // 当前块id
    uint32_t totalDist;  // 总距离数量
    
    //  新增：分块处理参数
    uint32_t blockSize;  // 每个分块的大小
    uint32_t numBlocks;  // 总的分块数量
    
    uint32_t tilingKey;
    GlobalTensor<float> gradGm;
    GlobalTensor<float> inputGm;
    GlobalTensor<float> pdistOutputGm;
    GlobalTensor<float> resultGm;
    
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInput;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGrad;
    TQue<QuePosition::VECCALC, BUFFER_NUM> tempQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> syncQueue;
    
    static constexpr int BUFFER_NUM = 2;
};

extern "C" __global__ __aicore__ void pdist_grad(GM_ADDR grad, GM_ADDR input, GM_ADDR pdist_output, GM_ADDR result, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    PdistGradKernel kernel;
    kernel.Init(grad, input, pdist_output, result,
                tiling_data.n, tiling_data.d, tiling_data.p, 
                tiling_data.totalDist, tiling_data.tilingKey, tiling_data.blocknum,
                tiling_data.blockSize);
    kernel.Process();
=======
extern "C" __global__ __aicore__ void pdist_grad(GM_ADDR grad, GM_ADDR input, GM_ADDR pdist_ouput, GM_ADDR result, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
>>>>>>> 41a96658ee2cc44167b5b073209842e137ceacde
}