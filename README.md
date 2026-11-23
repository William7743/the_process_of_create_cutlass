# 从一次乘加开始搭建 CuTe GEMM

目标：沿 NVIDIA CUTLASS v3.9.2 的 `examples/cute/tutorial/sgemm_1.cu`，在 RTX 3090 上从底层向上理解标量 FP32 GEMM。每个提交增加一个可解释的部件。

## 01 · 更新一个累加值

先读 [`real.hpp` 第 55–62 行](include/cute/numeric/real.hpp#L55-L62)：给定两个输入 a、b 和以前累加到的 c，把 a × b + c 写入 d。例如 a=2，b=3，c=5，得到 d=11。在矩阵乘里，这是给一个输出元素加上 K 方向的一项贡献。

再读 [`mma.hpp` 第 44–62 行](include/cute/arch/mma.hpp#L44-L62)：`UniversalFMA` 把这个动作包装成一个类型。四个长度为 1 的数组别名描述每个操作数需要一个值；声明别名本身没有分配四个数组。成员函数把工作交给对应数据类型的自由函数 `fma`。这里表达运算的语义，不承诺编译器一定生成某条机器指令。

为什么选它：[教程第 214 行](https://github.com/NVIDIA/cutlass/blob/v3.9.2/examples/cute/tutorial/sgemm_1.cu#L214) 调用三参数 `gemm`；[CuTe 的默认实现](https://github.com/NVIDIA/cutlass/blob/v3.9.2/include/cute/algorithm/gemm.hpp#L142-L160) 选用 `MMA_Atom<UniversalFMA<...>>`。本步只取最里面的 `UniversalFMA` 和实数运算，尚未放入 `MMA_Atom`。

第 01 次提交的状态：两份头文件均完整取自上游同名路径，保留文件头的 BSD-3-Clause 声明；此快照尚缺依赖头文件和运行入口，不能独立编译，也未在 GPU 上执行。

## 02 · 如何找到一个元素：Layout

一个线程要读 A(m,k)，首先必须知道这个元素的地址。`layout.hpp` 把坐标映射到线性偏移。例如形状 (128,8)、步长 (1,128) 的块中，(3,2) 对应偏移 259。改变布局就能改变存储顺序，无需改变乘加公式。
