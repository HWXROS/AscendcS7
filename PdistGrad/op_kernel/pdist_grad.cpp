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
        this->blockSize = blockSize;  
        this->numBlocks = (d + blockSize - 1) / blockSize;  
        this->gradGm.SetGlobalBuffer((__gm__ float*)grad, totalDist);
        this->inputGm.SetGlobalBuffer((__gm__ float*)input, n * d);
        this->pdistOutputGm.SetGlobalBuffer((__gm__ float*)pdist_output, totalDist);
        this->resultGm.SetGlobalBuffer((__gm__ float*)result, n * d);
//         if (GetBlockIdx()==0)
//         {
//             printf("blocknum %d, blockTileNum %d, numBlocks %d \n", this->blockNum, GetBlockNum(),this->numBlocks);
//             printf("N %d, D %d \n", this->n, this->d);
//         }
//         blocknum 1, blockTileNum 1, numBlocks 1 
//         N 3, D 32 
        

        const uint32_t BLOCK_BYTES = blockSize * sizeof(float);
        pipe.InitBuffer(inQueueInput1, BUFFER_NUM, BLOCK_BYTES * 2);   // 两个向量的一个分块
        pipe.InitBuffer(inQueueInput2, BUFFER_NUM, BLOCK_BYTES * 2);   // 两个向量的一个分块
        pipe.InitBuffer(gradInput1, BUFFER_NUM, BLOCK_BYTES * 2);   // 两个向量的一个分块
        pipe.InitBuffer(gradInput2, BUFFER_NUM, BLOCK_BYTES * 2);   // 两个向量的一个分块
        pipe.InitBuffer(QueueOutput1, BUFFER_NUM, BLOCK_BYTES * 2);   // 两个向量的一个分块
        pipe.InitBuffer(QueueOutput2, BUFFER_NUM, BLOCK_BYTES * 2);   // 两个向量的一个分块
        pipe.InitBuffer(gradBuf, BLOCK_BYTES );      // 中间结果
        pipe.InitBuffer(tempBuf, BLOCK_BYTES * 4);      // 中间结果

        // 初始化result为0
        InitResultToZero();
    }
    

    
    // 初始化result为0
    __aicore__ inline void InitResultToZero() {
        uint32_t elementsPerCore = (n * d + blockNum - 1) / blockNum;
        uint32_t start = elementsPerCore * blockId;
        uint32_t end = min(start + elementsPerCore, n * d);
        for (uint32_t elem = start; elem < end; elem += blockSize) {
            uint32_t remaining = end - elem;
            uint32_t currentBlockSize = min(blockSize, remaining);
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
        
    }
    
private:
    
    __aicore__ inline void ProcessOnePair(uint32_t distIdx) {
        uint32_t i, j;
        DistIdxToPair(distIdx, i, j);
        float gradValue = gradGm.GetValue(distIdx);
        if (abs(gradValue) < 1e-12f) return;
        float distance = pdistOutputGm.GetValue(distIdx);
        if (distance < 1e-8f) return;
        ComputeGradientByBlocks(i, j, gradValue, distance);
    }
    __aicore__ inline void DistIdxToPair(uint32_t idx, uint32_t& i, uint32_t& j) {
        i = 0;
        while (idx >= n - i - 1) {
            idx -= n - i - 1;
            i++;
        }
        j = i + idx + 1;
    }

    
    __aicore__ inline void Getscale(float distance, float gradValue, float& scale){
        if (p == 2.0f) {
                    scale = gradValue / distance;
                } 
        else if (p == 1.0f) {
                    scale = gradValue;
                }  
    }
    __aicore__ inline void ComputeGradientByBlocks(uint32_t i, uint32_t j, 
                                               float gradValue, float distance) {
    float scale = 1.0f;
    Getscale(distance, gradValue, scale);
    
    // 遍历所有分块计算梯度
    for (uint32_t block = 0; block < numBlocks; ++block) {
        uint32_t blockStart = block * blockSize;
        uint32_t currentBlockSize = (block == numBlocks - 1) ? (d - blockStart) : blockSize;
        
        // Copy IN
        {
            LocalTensor<float> vec_i_block = inQueueInput1.AllocTensor<float>();
            LocalTensor<float> vec_j_block = inQueueInput2.AllocTensor<float>();
            
            // 从全局内存加载输入数据
            SetAtomicNone();
            DataCopy(vec_i_block, inputGm[i * d + blockStart], currentBlockSize);
            DataCopy(vec_j_block, inputGm[j * d + blockStart], currentBlockSize);
            
            inQueueInput1.EnQue(vec_i_block);
            inQueueInput2.EnQue(vec_j_block);
        }
        
        // Compute
        {
            LocalTensor<float> vec_i_block = inQueueInput1.DeQue<float>();
            LocalTensor<float> vec_j_block = inQueueInput2.DeQue<float>();
            
            LocalTensor<float> diff = tempBuf.GetWithOffset<float>(0, currentBlockSize);
            LocalTensor<float> gradBlock = gradBuf.GetWithOffset<float>(0, currentBlockSize);
            
            // 计算梯度
            Sub(diff, vec_i_block, vec_j_block, currentBlockSize);
            Muls(gradBlock, diff, scale, currentBlockSize);
            
            // 准备输出张量
            LocalTensor<float> grad_i = QueueOutput1.AllocTensor<float>();
            LocalTensor<float> grad_j = QueueOutput2.AllocTensor<float>();
            
            // 复制梯度值（i 的梯度为正，j 的梯度为负）
            DataCopy(grad_i, gradBlock, currentBlockSize);
            Muls(grad_j, gradBlock, -1.0f, currentBlockSize); // j 的梯度是负的（修复：使用 Muls 替代 Neg）
    
            
            QueueOutput1.EnQue(grad_i);
            QueueOutput2.EnQue(grad_j);
            
            inQueueInput1.FreeTensor(vec_i_block);
            inQueueInput2.FreeTensor(vec_j_block);
        }
        
        // Copy OUT (原子累加)
        {
            LocalTensor<float> grad_i = QueueOutput1.DeQue<float>();
            LocalTensor<float> grad_j = QueueOutput2.DeQue<float>();
            
            // 对 i 向量进行原子累加
            SetAtomicNone();
            SetAtomicAdd<float>();
            DataCopy(resultGm[i * d + blockStart], grad_i, currentBlockSize);
            SetAtomicNone();
            
            // 对 j 向量进行原子累加
            SetAtomicNone();
            SetAtomicAdd<float>();
            DataCopy(resultGm[j * d + blockStart], grad_j, currentBlockSize);
            SetAtomicNone();
            
            QueueOutput1.FreeTensor(grad_i);
            QueueOutput2.FreeTensor(grad_j);
        }
    }
}
    

    uint32_t n;           // 向量数量
    uint32_t d;           // 向量维度
    float p;             // p值
    uint32_t blockNum; //核心数量
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
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInput1,inQueueInput2,gradInput1,gradInput2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> QueueOutput1,QueueOutput2;
    TBuf<TPosition::VECCALC> tempBuf,gradBuf;

    
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

