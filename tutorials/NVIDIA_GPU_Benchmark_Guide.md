#  NVIDIA GPU Benchmark 测试手册

**修订记录**

| Date      | Version | Author  | Description |
| --------- | ------- | ------- | ----------- |
| 3/22/2022 | 1.0     | Wei Yan | 初始版      |



## 1. 单节点测试

### 1.1基础性能测试

#### 1.1.1 P2P

本测试，使用CUDA Samples中提供的p2pBandwidthLatencyTest工具，测试多卡间带宽。结果会包含单向带宽p2p，双向带宽p2p，以及latency性能参数。

测试步骤：

1. 从NGC拉取测试用镜像，本示例中使用TensorFlow镜像。

   ```
   docker pull nvcr.io/nvidia/tensorflow:21.07-tf1-py3
   ```

2. 将CUDA Samples复制到本地用户目录

   ```
   git clone https://github.com/NVIDIA/cuda-samples
   ```

3. 启动容器，挂载本地CUDA Samples的目录

   ```
   nvidia-docker run -it --rm -v /your_local_dir/:/workspace
   nvcr.io/nvidia/tensorflow:21.07-tf1-py3
   ```

4. 在容器中，进入CUDA Samples目录

   ```
   cd cuda-samples/Samples/5_Domain_Specific/p2pBandwidthLatencyTest
   ```

5. 使用make编译测试命令

   ```
   sudo make
   ```

6. 开始测试

   ```
   ./p2pBandwidthLatencyTest
   ```

#### 1.1.2 Bandwidth

本测试，使用CUDA Samples中提供的bandwidthTest工具，可以测试主机到GPU（Host to device）复制带宽和GPU到GPU(device to device)复制带宽。

测试步骤：

1. 从NGC拉取测试用镜像，本示例中使用TensorFlow镜像

   ```
   docker pull nvcr.io/nvidia/tensorflow:21.07-tf1-py3
   ```

2. 将CUDA Samples复制到本地用户目录

   ```
   git clone https://github.com/NVIDIA/cuda-samples
   ```

3.  启动容器，挂载本地CUDA Samples的目录

   ```
   nvidia-docker run -it --rm -v /your_local_dir/:/workspace
   nvcr.io/nvidia/tensorflow:21.07-tf1-py3
   ```

4.  在容器中，进入CUDA Samples的目录

   ```
   cd cuda-samples/Samples/1_Utilities/bandwidthTest
   ```

5. 使用make编译测试命令

   ```
   sudo make
   ```

6. 开始测试

   ```
   ./bandwidthTest
   ```

   更多CUDA Spamples测试，请查阅[cuda-samples](https://github.com/NVIDIA/cuda-samples)

#### 1.1.3 GEMM

​	矩阵乘法是高性能计算中最常用到一类计算模型。无论在HPC领域，例如做FFT、卷积、相关、滤波 等，还是在 Deep Learning 领域，例如卷积层，全连接层等，其核心算法都直接或者可以转换为矩阵乘 法。

​	cuBLAS 是标准线性代数库 (standard basic linear algebra subroutines (BLAS)) 的 GPU 加速实现， 它支持 Level 1 (向量与向量运算) ，Level 2 (向量与矩阵运算) ，Level 3 (矩阵与矩阵运算) 级别的标准矩 阵运算。

​	GEMM（General matrix multiplication）是NVIDIA提供的二进制测试工具，利用cuBLAS库，通过随 机数进行矩阵乘运算，测试GPU的Peak TFLOPs。通过设定参数，GEMM可以测试不同数据类型。使用test_tool文件夹中的cublasMatmulBench工具进行测试。

1. 将cublasMatmulBench二进制文件复制到要测试的服务器上。

2. 为二进制文件赋予执行权限

   ```
   sudo chmod -R 777 cublasMatmulBench
   ```

3. 执行测试

   注意：该测试命令运行在单GPU上，默认调用GPU 0，如需测试全部GPU，可以使用Docker镜像挂载不同的GPU来测试。

   使用docker挂载单个GPU示例命令：

   ```
   nvidia-docker run -it --gpus '"device=1"' --rm -v
   /your_cublasMatmulBench_file_dir/:/workspace
   nvcr.io/nvidia/tensorflow:21.07-tf1-py3
   ```

   进入容器后，通过nvidia-smi查看，仅挂载了1个GPU。

   ```
   root@1c49a3fcece6:/workspace# nvidia-smi
   Thu Dec 30 08:02:32 2021
   +---------------------------------------------------------------------------
   --+
   | NVIDIA-SMI 470.57.02 Driver Version: 470.57.02 CUDA Version: 11.4
   |
   |-------------------------------+----------------------+--------------------
   --+
   | GPU Name Persistence-M| Bus-Id Disp.A | Volatile Uncorr.
   ECC |
   | Fan Temp Perf Pwr:Usage/Cap| Memory-Usage | GPU-Util Compute
   M. |
   | | | MIG
   M. |
   |===============================+======================+====================
   ==|
   | 0 NVIDIA A100-SXM... On | 00000000:0F:00.0 Off |
   0 |
   | N/A 30C P0 51W / 400W | 3MiB / 40536MiB | 0%
   Default |
   | | |
   Disabled |
   +-------------------------------+----------------------+--------------------
   --+
   +---------------------------------------------------------------------------
   --+
   | Processes:
   |
   | GPU GI CI PID Type Process name GPU
   Memory |
   | ID ID Usage
   |
   |===========================================================================
   ==|
   | No running processes found
   |
   +---------------------------------------------------------------------------
   --+
   
   ```

   GEMM测试命令如下：

   ```
   INT8:./cublasMatmulBench -P=bisb_imma -m=8192 -n=3456 -k=16384 -T=1000 -ta=1
   -B=0
   FP16:./cublasMatmulBench -P=hsh -m=12288 -n=9216 -k=32768 -T=1000 -tb=1 -B=0
   TF32:./cublasMatmulBench -P=sss_fast_tf32 -m=8192 -n=3456 -k=16384 -T=1000 -
   ta=1 -B=0
   FP32:./cublasMatmulBench -P=ddd -m=3456 -n=2048 -k=16384 -T=1000 -tb=1 -B=0
   FP64:./cublasMatmulBench -P=sss -m=3456 -n=2048 -k=16384 -T=1000 -tb=1 -B=0
   ```

   

#### 1.1.4 STREAM

NVIDIA 为 STREAM 基准测试提供优化的 CUDA 实现，用于测量单个 GPU 上的内存带宽。使用test_tool文件夹中的stream_test进行测试。 

测试步骤：

1. 将stream_test下载到测试服务器。

2. 为二进制文件赋予执行权限。

   ```
   sudo chmod -R 777 stream_test
   ```

3. 执行测试

   注意：该测试命令运行在单GPU上，通过调整-d参数后面的数字（0-7），就可以选择不同的GPU 执行。

   ```
   ./stream_test -d0 -n113246208 -r0
   ```

   

#### 1.1.5 HPL

​	Linpack 已成为全球最流行的测试高性能计算机系统浮点性能的基准。 用高性能计算机通过高斯消元 法求解n元一阶稠密线性代数方程来评价高性能计算机的浮点性能。

​	Linpack测试包括三类：Linpack 100、Linpack 1000和HPL。 NVIDIA NGC为 的 HPL benchmark测试提供了软件包，可在配备 NVIDIA GPU 的分布式内存计算机 上，基于 netlib HPL 基准测试，使用 Tensor Cores 以双精度（64 位）算法求解密集线性系统。

​	HPL-AI benchmark测试包含在 HPL benchmark测试中。 HPL-AI benchmark测试提供软件包以使用 Tensor Cores 在混合精度算法中解决（随机）密集线性系统。 

测试步骤：

1.  从NGC拉取测试用镜像，本示例中使用hpc镜像

   ```
   docker pull nvcr.io/nvidia/hpc-benchmark:20.10-hpl
   ```

2. 获取测试用data

   本测试需要使用HPL测试data，文件名为HPL-dgx-a100-1N.dat 。如有需要，请联系NVIDIA获 取。

3. 启动容器，挂载本地data的目录

   ```
   nvidia-docker run --privileged -it --rm -v $(pwd):/my-dat-files
   nvcr.io/nvidia/hpc-benchmarks:20.10-hpl
   ```

4. 在容器中，设置环境变量

   ```
   export UCX_TLS=all
   ```

   注意：NOTE: UCX_TLS=**rc_x** are set in the container, so for a single node without IB or down-state IB, there will be errors.

5. 开始测试

   ```
   mpirun --bind-to none -np 8 hpl.sh --config dgx-a100 --dat /my-datfiles/HPL-dgx-a100-1N.dat
   ```

   可以使用nvidia-smi监控GPU使用率。

#### 2.1.6 NCCL

NCCL是一种多GPU、多节点通信原语，针对 NVIDIA GPU 通信进行了优化。 NCCL allreduce 是一种 非常有效的方式来验证通过各种网络技术（例如 IB、RoCE、TCP/IP）的节点间 GPU 通信。

测试步骤：

1. 从NGC拉取测试用镜像，本示例中使用Pytorch镜像

   ```
   docker pull nvcr.io/nvidia/pytorch:21.06-py3
   ```

2.  启动容器，挂载本地用户的目录

   ```
   nvidia-docker run -it -v /home/your_user_name:/nccl
   nvcr.io/nvidia/pytorch:21.06-py3 bash
   ```

3.  在容器中，下载nccl-test并编译

   ```
   cd /nccl
   git clone https://github.com/nvidia/nccl-tests
   cd nccl-tests
   make MPI=1 MPI_HOME=/usr/local/mpi
   ```

4. 开始测试

   ```
   ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
   ```

   测试命令说明：以上测试命令，是在8个GPU上（-g 8）运行NCCL Test，测试数据包大小从8 Bytes 到 128MBytes。

   更多详细内容，请查阅[nccl-test]([NVIDIA/nccl-tests: NCCL Tests (github.com)](https://github.com/NVIDIA/nccl-tests))。

### 1.2 DL性能测试

#### 2.2.1 RestNet50

在计算机视觉领域，图像分类是非常重要的基本问题，是图像目标检测、图像分割、图像检索、视频理 解、物体跟踪、行为分析等其他高层视觉任务的基础，在实际场景中，有着广泛应用。

ResNet是近几年非常流行的卷积神经网络结构，其创造性提出的残差结构，一举在ILSVRC2015比赛中 取得冠军，并且获得计算机视觉顶级会议CVPR 2016的最佳论文。其中50层的网络结构(ResNet50)的效 果优化，备受学术界和工业界关注。

本测试基于TensorFlow框架，使用随机生成的合成数据集（您无需准备ImageNet数据集），即可快速 实现DL模型的Benchmark测试。

测试步骤：

1. 从NGC拉取测试用镜像，本示例中使用TensorFlow镜像

   ```
   docker pull nvcr.io/nvidia/tensorflow:21.07-tf1-py3
   ```

2. 启动容器

   ```
   nvidia-docker run -it --rm -v $(pwd):/work nvcr.io/nvidia/tensorflow:21.07-
   tf1-py3
   ```

3. 在容器中，进入cnn的目录

   ```
   cd nvidia-examples/cnn/
   ```

4. 开始测试

   参数说明：

   ​                    8节点:                 -np 8

   ​                    Batch-size大小 --batch_size 256

   ​                    数据精度            --precision fp16

   以下命令使用8节点，256 Batch-size，fp16测试

   ```
   mpiexec --allow-run-as-root -np 8 --bind-to socket python -u ./resnet.py --
   batch_size 256 --num_iter 1000 --precision fp16 --iter_unit batch --layers
   50
   ```

   以下命令使用8节点，256 Batch-size，fp32测试

   ```
   mpiexec --allow-run-as-root -np 8 --bind-to socket python -u ./resnet.py --
   batch_size 256 --num_iter 1000 --precision fp32 --iter_unit batch --layers
   50
   ```

   

