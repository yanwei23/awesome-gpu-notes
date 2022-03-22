#  NVIDIA GPU Performance Testing Guide



## 1. 单点测试

### 1.1基础性能测试

测试中需要用到cuda samples中的示例，需提前下载到测试服务器，代码地址:https://github.com/NVIDIA/cuda-samples。

#### 1.1.1 多卡间带宽测试

本测试使用CUDA Samples中提供的p2pBandwidthLatencyTest工具测试多卡间带宽。

测试步骤：

1. 启动容器，挂载本地CUDA Samples的目录，如果本地镜像不存在，将会自动从ngc拉取镜像。

   ```
   $ nvidia-docker run -it --rm -v /<local_dir>/cuda_samples:/workspace nvcr.io/nvidia/tensorflow:21.07-tf1-py3
   ```

2. 进入测试程序所在目录，并进行编译。

   ```
   $ cd /workspace/Samples/5_Domain_Specific/p2pBandwidthLatencyTest
   $ make
   ```

3. 开始测试

   ```
   $ ./p2pBandwidthLatencyTest
   ```

   测试结果包含单向带宽p2p，双向带宽p2p，以及latency性能参数。

   

#### 1.1.2 主机到GPU及GPU到GPU间带宽测试

本测试使用CUDA Samples中提供的bandwidthTest工具进行测试。

测试步骤：

1. 启动容器，挂载本地CUDA Samples的目录，如果本地镜像不存在，将会自动从ngc拉取镜像。

   ```
   $ nvidia-docker run -it --rm -v /<local_dir>/cuda_samples:/workspace nvcr.io/nvidia/tensorflow:21.07-tf1-py3
   ```

2. 进入测试程序所在目录，并进行编译。

   ```
   $ cd /workspace/Samples/1_Utilities/bandwidthTest
   $ make
   ```

6. 开始测试

   ```
   $ ./bandwidthTest
   ```

   更多CUDA Spamples测试，参考[cuda-samples](https://github.com/NVIDIA/cuda-samples)

#### 1.1.3 NCCL测试

NCCL是一种多GPU、多节点通信原语，针对 NVIDIA GPU 通信进行了优化。 NCCL allreduce 是一种非常有效的方式来验证通过各种网络技术（例如 IB、RoCE、TCP/IP）的节点间 GPU 通信。

测试步骤：

1. 启动容器，挂载本地用户的目录，如果本地镜像不存在，将会自动从ngc拉取镜像。

   ```
   $ nvidia-docker run -it -v /home/your_user_name:/nccl nvcr.io/nvidia/pytorch:21.06-py3 bash
   ```

2. 在容器中，下载nccl-test并编译

   ```
   $ cd /nccl
   $ git clone https://github.com/nvidia/nccl-tests
   $ cd nccl-tests
   $ make MPI=1 MPI_HOME=/usr/local/mpi
   ```

3. 开始测试

   ```
   $ ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
   ```

   测试命令说明：-g 指定GPU数量，-g 8 即是在8个GPU上运行NCCL Test

   ​                          测试数据包大小从8 Bytes 到 128MBytes。

   更多详细内容，请查阅[nccl-test]([NVIDIA/nccl-tests: NCCL Tests (github.com)](https://github.com/NVIDIA/nccl-tests))。

### 1.2 DL性能测试

#### 2.2.1 ResNet50

本测试基于TensorFlow框架，使用随机生成的合成数据集，通过resnet50网络进行测试。

测试步骤：

1. 启动容器，挂载本地工作路径。如果本地镜像不存在，将会自动从ngc拉取镜像。

   ```
   $ nvidia-docker run -it --rm -v $(pwd):/work nvcr.io/nvidia/tensorflow:21.07-tf1-py3
   ```

2. 进入测试程序目录，开始测试

   参数说明：

   ​                    8节点:                 -np 8

   ​                    Batch-size大小 --batch_size 256

   ​                    数据精度            --precision fp16

   以下命令使用8节点，256 Batch-size，fp16测试

   ```
   $ cd nvidia-examples/cnn/
   $ mpiexec --allow-run-as-root -np 8 --bind-to socket python -u ./resnet.py --batch_size 256 --num_iter 1000 --precision fp16 --iter_unit batch --layers 50
   ```

   参数说明：

   ​                    8节点:                 -np 8

   ​                    Batch-size大小 --batch_size 256

   ​                    数据精度            --precision fp16

