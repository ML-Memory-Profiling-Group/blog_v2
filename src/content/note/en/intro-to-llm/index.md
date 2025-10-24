---
title: 'Introduction to LLM Profiling'
timestamp: 2025-10-24 00:00:00+00:00
series: LLM
tags: [GPU]
description: 'A profiling tutorial for Nvidia GPUs with two different GPT-2 workflow'
---

# Introduction

# The Computational Core of Modern LLMs
A prerequisite for effective performance analysis is a foundational understanding of the target software's core working mechanisms. To establish a solid baseline for our performance expectations, this section will detail the internal workings of the GPT-2 architecture as implemented in our two C++ codebases: the Eigen-Optimized Kernel [LLM-Eigen](https://github.com/zhangpiu/llm.cpp.git) and the CCCL-Accelerated Engine [LLM-CCCL](https://github.com/gevtushenko/llm.c.git) both originated from legendary Andrej Karpathy’s [llm.c](https://github.com/karpathy/llm.c.git). We will illustrate how architectural and implementation choices shape the runtime behavior and performance characteristics of transformer-based models. This foundational understanding will help establish clear performance expectations and guide the deeper profiling and analysis discussed in later sections. Readers already familiar with GPT-2 internals and model execution pipelines may choose to skim this section and proceed directly to the detailed performance investigation.
To begin, let’s focus on the training forward pass while ignoring batching for simplicity, as depicted in the first layer of the GPT-2 architecture diagram.
![First layer of GPT2](gpt2-layer.png)

The model begins by converting the raw input text into a format suitable for computation, by breaking down into discrete tokens. These tokens are then mapped to continuous, high-dimensional vectors known as embeddings. This process, often handled via a lookup table, transforms sparse token IDs into dense, meaningful input vectors. The final input vector that enters the main transformer block combines these token embeddings with positional encodings. For both implementations, LLM-Eigen and LLM-CCCL, the main entry point is the **train_gpt2.cu** file. The training loop orchestrates the overall training process. In each iteration of this loop, a training sample and its corresponding target are loaded from the dataset, after which the forward pass of the model is executed. This forward pass forms the computational backbone of the training process, where most of the profiling and performance analysis will be focused in subsequent sections.

The code begins by allocating memory for activations, the intermediate tensors produced during forward and backward passes. The two implementations differ significantly in how they handle these allocations. The LLM-CCCL version allocates all activations in a single contiguous memory block, which reduces fragmentation and allows for efficient GPU memory access and management. In contrast, the LLM-Eigen implementation uses lazy allocation through the Eigen library’s LazyAllocate mechanism, which allocates memory on demand as tensors are created. While lazy allocation provides greater flexibility when working with varying tensor shapes and sizes, for example, during inference with dynamic sequence lengths, it may not yield the most optimized implementation for fixed-size training, where a bulk contiguous buffer is typically more efficient.

The training cycle begins with the entry into the Model's Forward Function. As illustrated in the architecture diagram , the initial processing steps are consistent: the input sequence is first tokenized, and the resulting token embeddings are summed with the positional encodings. This resultant vector, forms the initial input vector for the first transformer layer. The model then iterates through all transformer layers in sequence. Except for the first layer, the output of the preceding layer serves as the input to the next layer in the sequence.

A subtle but important difference lies in how the core operations within each transformer layer are structured and executed. In the LLM-Eigen version, each layer is encapsulated as a C++ class, with its constituent blocks implemented as member functions or objects of that class. In contrast, the CCCL version utilizes a flat, procedural execution of the blocks in every layer. Another key distinction lies in the implementation of sub-blocks within each transformer layer. The LLM-Eigen version adopts a mathematical operation-by-operation approach. Each core step of the calculation is mapped directly to an Eigen operation which, in turn, results in multiple, distinct kernel launches on the device. The LLM-CCCL codebase employs a more traditional high-performance strategy, where a single, fused kernel performs the entire computation of a sub-block. All these implementation choices have a direct impact on performance, and their effects will be analyzed in detail in the performance profiling section later in this blog.

The table below highlights the specific code sections corresponding to each implementation choice in both versions, providing an easy reference for readers to correlate source code with the architectural flow shown in the accompanying figure. Together, the table and figure serve as a practical guide to understanding how a modern LLM is structured and executed at a low level.

This reference will also be valuable in the next section, where we develop the roofline performance model for each of these blocks to quantify and compare their computational efficiency and memory behavior.

| Sub Blocks | LLM-Eigen | LLM-CCCL |
|------------|-----------|----------|
|Model Forward Function|ForwardGPU @ gpt.hpp|gpt2_forward @ train_gpt2.cu|
|Token Embedding & Positional Embedding|__Forward @ gpt.hpp|encoder_forward @ train_gpt2.cu|
|LLM Layer|Block::Forward @ gpt.hpp|encoder_forward @ train_gpt2.cu|
|LayerNorm|nn::LayerNorm::Forward @ nn.hpp|layernorm_forward @ train_gpt2.cu(layernorm_forward_kernel3)|
|QKV Linear Projection|CausalSelfAttention::Forward @ gpt.hpp(c_attn_->Forward)|matmul_forward_cublaslt @ train_gpt2.cu(cublasLtMatmul)|
|Self Attention: QKT|nn::MatMul::Forward @ gpt.hpp|attention_forward @ train_gpt2.cu(cublasSgemmStridedBatched)|
|Self Attention: Softmax|nn::Softmax::Forward @ gpt.hpp|softmax_forward_kernel5 @ train_gpt2.cu|
|Self Attention: Value Matmul|nn::Softmax::Forward @ gpt.hpp|attention_forward @ train_gpt2.cu(cublasSgemmStridedBatched)|
|O Linear Projection|nn::MatMul::Forward @ gpt.hpp|matmul_forward_cublaslt @ train_gpt2.cu(cublasLtMatmul)|
|Residual|nn::Residual::Forward @ gpt.hpp|residual_forward @ train_gpt2.cu|
|FeedForward: MLP1 & MLP2|MLP::Forward @ gpt.hpp nn::Linear::Forward|matmul_forward_cublaslt @ train_gpt2.cu(cublasLtMatmul)|
|FeedForward: GeLU|nn::NewGELU::Forward @ gpt.hpp|gelu_forward @ train_gpt2.cu|

# Basics of GPU

# CUPTI

CUPTI is a set of API that enables developers to both retrieve hardware counters from NVidia GPUs and trace the host-side activities on CUDA. It serves as the foundation of NSight Compute, the official GPU profiler provided by NVidia. With CUPTI, independent developers can develop customized profilers that leverage the same sets of metrics and derive their own specialized insights through custom data processing

In the big picture, CUPTI has two key functionalities:

* Tracing: collecting host-side activities, like kernel launches and memset, etc.  
* Profiling: collecting hardware counters and other derived metrics like throughput.

It can also be divided into multiple sets by the way it collects data, including

* the Activity API,  
* the Callback API,  
* the Host Profiling API,  
* the Range Profiling API,  
* the PC Sampling API,  
* the SASS Metric API,  
* the PM Sampling API,  
* the Checkpoint API,  
* the Profiling API,

For this blog, we built a GPU profiler on top of the Activities API and Range Profiling API to trace and profile. We won’t talk about the details about how our profiler is built, but rather we will focus on the performance data collected. In case you are interested, the profiler is available here(github\_link) and the corresponding tutorial in detail is also available here(blog\_link). Simply speaking, our profiler is able to separate code into logical blocks called “range” by wrapping the code with push and pop range functions, which defines the range we are interested in and would like to collect data from. Here is a sample code of how we wrapped and timed the range:

```cpp
  GmpProfiler::getInstance()->pushRange("MLP", GmpProfileType::CONCURRENT_KERNEL);
  mlp_->Forward(ln2_y_2d_const, mlp_y_2d)
  GmpProfiler::getInstance()->popRange("MLP", GmpProfileType::CONCURRENT_KERNEL);
```

GmpProfiler::getInstance()-\>pushRange/popRange is the API of our profiler that collects both traces and metrics and defines the range. GMP\_TIMED is simply a macro to use C++ chrono to get the CPU time spent by the wrapped portion of the code and this is where the wall-clock time comes from.

All the activities records and metrics collected will be grouped by range name and accumulated or averaged among all the kernels’ data within the range, so that we can understand how each phase of the LLM performs.

# Performance Analysis

## Roofline Performance
| SUB BLOCKS | NUM ELEMENTS | TOTAL SIZE (MB)|
|------------|--------------|----------------|
|Input | B * T * C | 0.75 |
|Layer Norm | B * L * T * C | 2.25 |
|Q, K, V | B * L * T * 3C | 6.75 |
|SoftMAX(QKT) | B * L * H * T * T | 2.25 |
|O | B * L * T * C | 2.25 |
|Residual | B * L * T * C | 2.25 |
|MLP1 | B * L * T * 4C | 9 |
|MLP GeLU | B * L * T * 4C | 9 |
|MLP2 | B * L * T * C | 2.25 |

## Kernel Invocations

Kernel launch is where CUDA assigns computation tasks to the GPU. The number of kernels and the size of blocks and grids can produce profound impact on system performance. Ideally, each kernel should have enough blocks and threads so that it doesn’t under utilize the compute resources. On the other hand, too many blocks, threads or kernel launches themselves will accumulate overheads and severely hurt the overall performance. In this section, we will see how the two implementations differ and why they differ. In later sections, we will discuss how these differences impact the performance

<center>Number of Kernels In The Ranges</center>
| Framework / Range         | attention | ln1 | ln2  | mlp | residual1 | residual2 |
|----------------------------|------------|----------|-----|-----|-----|------------|
| **Eigen**                 | 440        | 2   | 3   | 5   | 1          | 1          |
| **CCCL**                  | 8          | 1   | 1   | 4   | 1          | 1          |

If we compare the two implementations in forward process, it is obvious that the Eigen version launches more kernels, especially in the attention. In the CCCL version, it is implemented in a way that all the computation is fused in one global kernel. Only big ranges, like mlp and attention, will employ some helper device kernels. Whereas the kernel launches in Eigen are not explicit. In other words, the number of kernels and their grid/block sizes are dependent on the implementation of Eigen. It is not guaranteed that each assignment of Eigen will generate only one kernel, and that’s why generally the Eigen version launches more kernels than the CCCL version. This is the reason why most ranges of Eigen llm.cpp launch more kernels than the CCCL one.

However, this still doesn’t explain why the Eigen version of llm.cpp shows such an enormous attention range with 440 kernel launches—far more than any other range. The primary cause lies in the inefficient use of nested for loops within the implementation.Below is a pseudo-code example illustrating how the attention module is structured in Eigen’s llm.cpp:

```cpp
for (int b = 0; b < B; ++b)  
{  
    for (int h = 0; h < NH; ++h)  
    {  
        // Calculate Q K V

        // Calculate QK^T

        // softmax

        // att * V  
    }  
}  
```

This piece of code is looping over batch size and number of heads. There are two matrix multiplications and softmax in each iteration, which will produce quite a lot of kernels. With B=4 and NH=12, all these kernels are repeated 48 times, so no surprise so many kernels are launched. This exemplifies a pitfall of GPU programming. It is common and fine to use for loops when we write programs for CPU, but the misuse of for loops on GPU programs can heavily downgrade the performance. We will discuss the performance drop in the next section.

<center>Grid And Block Size Statistics</center>
|  | Eigen | CCCL |
| :---- | :---- | :---- |
| min | 1 | 16 |
| max | 3144 | 1536 |
| mean | 2.741594966 | 158.9488133 |
| median | 4 | 320 |
| avg warp/block | 23.51575188 | 8.282119391 |

Another key aspect to understand when analyzing GPU kernels is the grid size and block size. We've included the statistics for all implementations. The results clearly show the Eigen implementation tends to launch most kernels with only a handful of blocks (often single-digit), while the CCCL version consistently launches around 160 blocks on average. This difference has major implications for GPU utilization. The Eigen llm.cpp kernels are, in effect, severely underutilizing the GPU's compute resources. Our tests were conducted on an NVIDIA A100, which features 108 streaming multiprocessors (SMs). Ignoring stalls from data dependencies and assuming a single active CUDA stream, we can reason that since a block cannot span across multiple SMs, we need at least 108 blocks to fully occupy all SMs—one block per SM. Our estimation of SM utilization is:

$$SmUtilization = \frac{\max(DeviceSmNum, BlockSize)}{DeviceSmNum} \times 100\%$$

This formula is based on the assumption of single device and no parallel kernel execution. The Eigen llm.cpp forward kernels only launch about 2.7 blocks per launch on average, which means that roughly 2% of the SMs are actually being used—an astonishingly low figure for such a powerful GPU. It’s worth noting that we assume the scheduler assigns blocks to idle SMs first, rather than sharing SMs among active blocks, since this policy maximizes utilization. NVIDIA’s exact block scheduling policy isn’t publicly documented, so this conclusion is based on empirical observation and reasonable inference. Regardless of the precise scheduling details, the number of blocks required for full utilization must be at least 108, so our conclusion about underutilization remains valid.

## Wall Clock Time and GPU execution Time

Time consumption is one of the key metrics of any type of program. In this blog we will focus on the wall clock time and the GPU time of the two implementations.

* The GPU execution time is the time GPU takes to finish all the calculation. It is collected from hardware counters per kernel through CUPTI. For a range, we add the gpu\_\_time\_duration.max of all the kernels within it.
* The wall clock time is the time from the start to the end of executing a piece of code on the CPU. We use C++ std::chrono::high\_resolution\_clock::now() to wrap the code snippet and get the duration through subtracting the two timestamps. This time is measured on the CPU side, so it will include all the time spent by the code wrapped, including GPU execution time, launch overhead, any housekeeping operations like makespan, makeMatrix, synchronized cuda memory copy/memset, etc.

Note that by default the kernel launch is asynchronous and launching a kernel will only push it into a queue but not execute it. With CUPTI range profiling enabled, all kernels will be executed synchronously. That’s why our wall clock time includes the GPU execution time.

Below are the wall clock time(in microseconds) and GPU time(in nanoseconds) of all the ranges and their ratio. A huge overall performance gap between Eigen and CCCL implementation are presented and many factors contribute to these gaps.

![][forward-gpu-time]![][forward-gpu-time-ratio] 

Let's start from the GPU time. Without considering CPU side, the gap between the two versions is still enormous. The biggest contributors to the GPU time are the attention and mlp ranges, which is as expected because according to the roofline calculation, these two ranges did most FLOPs and MOPs. However, if we compare the ratio of GPU time between the two versions, we can observe that the attention range of the Eigen llm.cpp significantly outweighs the mlp range, whereas in CCCL llm.cpp, these ranges are relatively equivalent. According to the prior section, we know that huge amounts of small kernels are launched within the attention of Eigen llm.cpp. This causes several problems:
* Reduced locality on L1 because L1 is flushed between kernels.
* Poor latency hiding due to the shortage of blocks and short kernels.
* Low SM utilization because of the lack of blocks to be assigned to SMs.

Another noteworthy point is the developer of the CCCL llm.cpp implementation applied several optimizations to improve cache efficiency — for instance, using cache streaming to allow one-time data to bypass the cache, and employing reverse iteration to increase cache hits at the tail of arrays. In contrast, the Eigen version lacks such low-level optimizations, at least from the user side. As a result, the CCCL version achieves higher cache hit rates and fewer dram accesses, which directly contributes to its shorter execution time.

![][forward-wallclock-time]![][forward-wallclock-time-ratio]

On the other side, the gap of the wall clock time enlarges, suggesting that there are more factors outside GPU that further drops down the overall performance. We suggest that the additional dropdown is probably dorminated by launch overhead. To explain the difference, we conducted an experiment using a simple helloworld CUDA program. In this program, we launched 440 kernels with minimum FLOPS and MFLOPS and the average wall clock time we got is 12.4 microseconds per kernel, which should mostly be launch overhead. In comparison, we calculated the average gaps between kernels to estimate the launch overheads of the two llm.cpp using this formula:

$$AvgLaunchOverhead = \frac{RangeWallClockTime - GpuExecutionTime}{KernelNum}$$

The result is presented here:

<center>Average Gap Between Wall Clock Time And GPU Time</center>
| Layer         | Eigen Avg Gap (µs) | CCCL Avg Gap (µs) |
|----------------|--------------------------------|--------------------------------|
| ln1            | 5.136                          | 14.544                         |
| attention      | 6.669                          | 5.053                          |
| residual1      | 14.288                         | 13.128                         |
| ln2            | 3.747                          | 14.064                         |
| feed_forward   | 90.354                         | 9.172                          |
| residual2      | 15.064                         | 13.840                         |

We can see that the avg gap mostly lies between 3~15µs, which matches the maginitude of approximate launch overheads from helloworld. This indicates that the extra wall clock time of the two llm.cpp is mostly launch overheads. Other than this, there are other minor possibilities that can contribute to the wall clock time:
* Allocation and release of resources, e.g. registers and shared memory, especially for the attention of Eigen.
* Synchronized Host-to-Device or Device-to-Host memory transfer or memset. 
* Execution time of CPU instructions.
* Overheads for library to choose appropriate kernels.

## SASS Instructions

SASS (Streaming Assembler) is the low-level assembly language executed by NVIDIA GPUs. It’s the final compiled form of CUDA kernels. CUPTI allows us to collect all kinds of SASS, but in this blog, we will focus on the global load/store instructions and the bytes it reads/writes. Here is the sample SASS instruction data of the residual range:

| metrics | Eigen | CCCL |
| :---- | :---- | :---- |
| smsp\_\_sass\_data\_bytes\_mem\_global\_op\_ld.sum | 1572864 | 1572864 |
| smsp\_\_sass\_data\_bytes\_mem\_global\_op\_st.sum | 786432 | 786432 |
| smsp\_\_sass\_inst\_executed\_op\_global\_ld.sum | 3072 | 12288 |
| smsp\_\_sass\_inst\_executed\_op\_global\_st.sum | 1536 | 6144 |

* smsp\_\_sass\_data\_bytes\_mem\_global\_op\_ld.sum is the total number of global load **warp** instructions issued. Note that this doesn’t include atomic or shared loads,which are collected in other metrics. 
* smsp\_\_sass\_data\_bytes\_mem\_global\_op\_ld.sum represents the actual data loaded by the SASS instructions. 

Store instructions are similar to the load instructions. We choose residual as an example because it is relatively straightforward and only contains an element-wise add operation. The GPU should load two input matrices and store the output matrix. That’s why there are 2x load instructions and bytes compared to stores. You may also notice that even though both implementations load/store the same amount of data, the Eigen version executed only ¼ instructions of the CCCL version. This is because the Eigen version employs vectorized loads for contiguous elements so that each global load will load 4 floats instead of 1 float. We can calculate it through following this formula:

$$AvgFloatsLoadedPerThread = \frac{SassBytesLoaded}{LoadSassInstIssued \times 32 \times 4}$$

We divide 32 because the issued SASS instructions are counted in warps. The 4 comes from 4 bytes per float. The result of the Eigen version is 4 floats per load. For the CCCL version, this number is reduced to 1 float per load. The vectorization is one optimization Eigen implicitly does for loads and stores automatically, which can reduce redundant instructions and issue overheads.

## L1, L2 and dram accesses

When SASS loads and stores are executed in the thread, they will be coalesced with other instructions executed by other threads within the warp and sent to L1. If the request missed,L1 will forward the request to L2. If it still misses, L2 will send requests to the dram in sectors. Here are the metrics we are interested and we will still show the residual range as an example:  

* l1tex\_\_t\_requests\_pipe\_lsu\_mem\_global\_op\_ld.sum: approximately the global load requests L1 cache received from the warps. The “lsu” implies that the requests are from the load store unit. The “approximately” means there might be requests other than global loads, like LDSTS instructions, but this is not so important and is beyond the scope of this blog. In most cases, you can find that it ballparkly matches the number of SASS load requests.  
* l1tex\_\_t\_sectors\_pipe\_lsu\_mem\_global\_op\_ld.sum is the number of sectors accessed by the requests received by L1 cache. In general this metric should be greater or equal to smsp\_\_sass\_data\_bytes\_mem\_global\_op\_ld.sum/32. Since the warps access contiguous and 32-aligned addresses in residual range, it exactly matches that result.   
* lts\_\_t\_requests\_srcunit\_tex\_op\_read.sum: the requests L2 cache received from L1.  
* lts\_\_t\_sectors\_srcunit\_tex\_op\_read.sum: the sectors accessed by the L2 requests from L1. Each request can contain 1\~4 sectors. This metric also represents how many sectors L1 missed.  
* dram\_\_sectors\_read.sum: sectors requested from L2 because of L2 misses.Note that though dram sends the data in bursts, the unit of these metrics is 32 byte sectors, so these metrics should be the actual bytes loaded divided by 32\.

![][residual-accesses]

In general, from L1 to L2 to dram, the sector metrics should gradually reduce. The higher the hit rates, the more they reduce. Here we can see L1 sector loads and L2 sector loads are the same. This is because all the addresses in residual will only be accessed once, so the hit rate is 0%. All the sectors being accessed in L1 are forwarded to L2.  Previously we mentioned that the Eigen llm.cpp is utilizing vectorized load, and that’s why the L1 requests of Eigen are relatively low compared to the CCCL version. There are also different dram sector reads between two implementations. This is probably because of L2 partitions or the activations that remain in L2 since L2 will not be flushed between kernel launches.

## Dram throughput

Finally, after requests have been filtered through L1 and L2, they reach dram, whose bandwidth greatly affects the overall performance of the system. CUPTI provides dram\_\_throughput.avg.pct\_of\_peak\_sustained\_elapsed, a percentage showing how much of theoretical sustained peak throughput one kernel can use, but this metrics only measures per kernel throughput. If we calculate the average throughput through adding all the metrics in range and divide by number of kernels in range, in some extreme cases, it may show misleading throughput because it loses the information of time. For example, if we have 1 kernel that heavily utilizes 100% throughput for an hour and 99 kernels use 0% in just 1 second, we will get an average usage of 1%, which looks pretty off. Therefore, instead of directly averaging the throughput metrics provided by CUPTI, we calculate the overall throughput by doing

$$DramThroughput = \frac{(DramReadSectors + DramWriteSectors) \times 32}{GpuTime}$$

Here is the data we produced:

![][dram-throughput]  

From the chart, we can find that generally the CCCL version consumes more dram throughput than the Eigen one. Previously we talked about the low grid size of the Eigen version. If the kernel is short of blocks, it will use few SMs, and the number of warp instructions issued per cycle will be limited because most SMs are not active. Remember the average grid size for Eigen implementation is 2.7. This makes most of the SM inactive, not being able to issue store or load commands and leave the remaining throughput wasted. Another reason might be  the GPU time of the range. If we refer back to the prior section, we can find that the CCCL version takes less than 1/10 GPU time of the Eigen ones. Our equation indicates that the denominator is the GPU time. With the same amount of dram loads and stores, the bandwidth will be multiple times higher if the time is as short as that. The reduced time of the CCCL llm.cpp indicates a better usage of dram bandwidth over leaving the bandwidth wasted for a long period of time.

Furthermore, we can see that in both implementations, the layer norms barely accessed the dram. This is expected because the calculation of the norms doesn't involve any parameters. All it needs is to load the previous activations and store the result norm. As L2 will not be flushed across kernels, the activation produced by the previous range should still reside in the L2. Therefore even if there will be SASS  loads and L1 requests, these accesses will be filtered out by L2 and keep the dram intact. That's another reason to explain layer norms use such a little throughput in both implementations other than the grid size.  

[kernel-num]: <kernel-num.png>
[forward-wallclock-time]: <forward-wallclock-time.png>
[forward-wallclock-time-ratio]: <forward-wallclock-time-ratio.png>
[kernel-num-ratio]: <kernel-num-ratio.png>
[forward-gpu-time]: <forward-gpu-time.png>
[forward-gpu-time-ratio]: <forward-gpu-time-ratio.png>
[residual-accesses]: <residual-accesses.png>
[dram-throughput]: <dram-throughput.png>