# 从一次乘加开始搭建 CuTe GEMM

目标：沿 NVIDIA CUTLASS v3.9.2 的 `examples/cute/tutorial/sgemm_1.cu`，在 RTX 3090 上从底层向上理解标量 FP32 GEMM。每个提交增加一个可解释的部件。

## 01 · 更新一个累加值

先读 [`real.hpp` 第 55–62 行](include/cute/numeric/real.hpp#L55-L62)：给定两个输入 a、b 和以前累加到的 c，把 a × b + c 写入 d。例如 a=2，b=3，c=5，得到 d=11。在矩阵乘里，这是给一个输出元素加上 K 方向的一项贡献。

再读 [`mma.hpp` 第 44–62 行](include/cute/arch/mma.hpp#L44-L62)：`UniversalFMA` 把这个动作包装成一个类型。四个长度为 1 的数组别名描述每个操作数需要一个值；声明别名本身没有分配四个数组。成员函数把工作交给对应数据类型的自由函数 `fma`。这里表达运算的语义，不承诺编译器一定生成某条机器指令。

为什么选它：[教程第 214 行](https://github.com/NVIDIA/cutlass/blob/v3.9.2/examples/cute/tutorial/sgemm_1.cu#L214) 调用三参数 `gemm`；[CuTe 的默认实现](https://github.com/NVIDIA/cutlass/blob/v3.9.2/include/cute/algorithm/gemm.hpp#L142-L160) 选用 `MMA_Atom<UniversalFMA<...>>`。本步只取最里面的 `UniversalFMA` 和实数运算，尚未放入 `MMA_Atom`。

第 01 次提交的第 01 次提交的第 01 次提交的第 01 次提交的状态：两份头文件均完整取自上游同名路径，保留文件头的 BSD-3-Clause 声明；此快照尚缺依赖头文件和运行入口，不能独立编译，也未在 GPU 上执行。

## 02 · 如何找到一个元素：Layout

一个线程要读 A(m,k)，首先必须知道这个元素的地址。`layout.hpp` 把坐标映射到线性偏移。例如形状 (128,8)、步长 (1,128) 的块中，(3,2) 对应偏移 259。改变布局就能改变存储顺序，无需改变乘加公式。

## 03 · 如何把一块数据分给线程：Tensor 与分区

`tensor_impl.hpp` 把存储与布局组成 Tensor 视图。`local_tile` 选择线程块负责的区域，`local_partition` 选择一个线程负责的元素；这两步只生成视图。后续 `copy` 才实际搬运数据。这样同一份线程分工既可用于全局内存的源，也可用于共享内存的目的地。

## 04 · 如何把乘加推广到整个分块：Atom 与算法

一个标量乘加只更新一个值。`mma_atom.hpp` 把一次运算包装为算法可调用的 Atom；`algorithm/gemm.hpp` 根据张量的形状和秩组织多次运算。教程选用 UniversalFMA。此处并不要求调用 Tensor Core。

## 05 · 如何让以上组件运行：补齐依赖并加入原版示例

`examples/cute/tutorial/sgemm_1.cu` 把以上组件装配成可启动的 CUDA kernel：分配输出块，按 K 块搬入 A、B，累加，最后写回 C。依赖集合包含被公共头文件间接引入的其他架构定义，不代表 3090 执行了这些架构的指令。

从仓库根目录编译（CUDA 12.4、GCC 12.1，主机编译器与运行时库需匹配）：

```bash
nvcc -ccbin g++ -std=c++17 -arch=sm_86 -O3 --expt-relaxed-constexpr -Iinclude -Itools/util/include examples/cute/tutorial/sgemm_1.cu -o sgemm
./sgemm 256 256 64 N T
```

前四个提交用于分层阅读；本提交包括实测依赖清单中的 141 个官方文件。教程的运行输出是冒烟检查，不是逐元素正确性证明。这条路径演示 CuTe 的标量 GEMM，不经过 `GemmUniversalAdapter`。

## 06 · 把线程分工和硬件指令接成一个主循环

对比上一章：当乘加换成一组线程共同完成的 Tensor Core 指令，怎样保留分块、搬运和写回的组织方式？先读[官方测试采用的配置](test/unit/gemm/device/default_gemm_configuration.hpp#L157)：128×128×32 的线程块计算块、128 个线程、三阶段异步拷贝主循环。第 169–172 行用 TiledMMA 安排线程协作，第 193 行附近把搬运和计算组装为 CollectiveMainloop；尾处理另外组装。

单条硬件矩阵操作在 [mma_sm80.hpp](include/cute/arch/mma_sm80.hpp#L158)，第 173 行给出对应 PTX。`mma_atom.hpp` 等头在标量例子的依赖中已经出现过；这次配置实际选择了 FP16 Tensor Core 操作。

## 07 · 给主循环与尾处理配上 GPU 入口和 CPU 接口

先读[官方用例第 57–71 行](test/unit/gemm/device/sm80_gemm_f16_f16_f32_tensor_op_f32.cu#L57)：取上一章的配置，把主循环和尾处理交给 `GemmUniversal`，再把 kernel 交给 `GemmUniversalAdapter`。主机现在有了启动入口。`gemm_universal.hpp` 汇集特化；此处通用 kernel 的实现位于 [sm70_gemm.hpp](include/cutlass/gemm/kernel/sm70_gemm.hpp#L55)，按 `KernelMultistage` 调度策略匹配，文件名不意味着在 3090 上执行 SM70 矩阵指令。

## 08 · 从空目录构建五层 GEMM 并验证四种布局

从 `nvcc -M` 收集三个编译单元的仓库内依赖，把[清单中的 471 个官方文件](verification/sm86-files.txt)复制到空目录后独立编译。RTX 3090、CUDA 12.4、GCC 12.1、GoogleTest 1.14.0；[四个测试均通过](verification/sm86-tests.txt)。`cuobjdump --dump-sass` 在二进制中检出 `HMMA.16816.F32`。471 包含测试框架适配、参考计算以及公共头间接带入的实现；不代表五层设计必须手写 471 个文件。

复现：在仓库根目录用 Bash 执行。`G` 指向解包的 GoogleTest 的 `googletest/`（含 src 和 include）；把本机工具链路径换成实际路径。

```bash
set -eu
W=$(mktemp -d); G=/path/to/googletest-1.14.0/googletest
CUDA=/usr/local/cuda-12.4; GCC=/usr/local/gcc-12.1.0
export PATH="$CUDA/bin:$GCC/bin:$PATH" LD_LIBRARY_PATH="$GCC/lib64:$CUDA/lib64:${LD_LIBRARY_PATH:-}"
TEST=test/unit/gemm/device/sm80_gemm_f16_f16_f32_tensor_op_f32.cu
F=(-std=c++17 -arch=sm_86 -ccbin "$GCC/bin/g++" --expt-relaxed-constexpr --expt-extended-lambda -Iinclude -Itools/util/include -Itest/unit -Itest/unit/common -I"$G/include")
nvcc "${F[@]}" -O2 -c "$TEST" -o "$W/gemm.o"
g++ -std=c++17 -O2 -pthread -I"$G" -I"$G/include" -c "$G/src/gtest-all.cc" -o "$W/gtest.o"
for t in test/unit/test_unit.cpp test/unit/common/filter_architecture.cpp; do nvcc "${F[@]}" -O2 -c "$t" -o "$W/$(basename "$t").o"; done
nvcc -arch=sm_86 -ccbin "$GCC/bin/g++" -Xcompiler=-pthread "$W"/{gemm,gtest,test_unit.cpp,filter_architecture.cpp}.o -o "$W/sm86-tests"
OMP_NUM_THREADS=4 "$W/sm86-tests"
```
