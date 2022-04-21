# DEEPSTREAM Python Demo

## 环境准备

### 1.设置GPU环境

参考[setup-gpu-env.md](https://github.com/yanwei23/awesome-gpu-notes/blob/main/tutorials/setup-gpu-env.md)

### 2. 从NGC下载deepstream对应的镜像并启动容器。

NGC链接：[DeepStream | NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream)

1. 进入NGC页面，点击右上角Pull Tag。基本的使用测试可选择devel分支的镜像。此时拉镜像的命令就会复制倒剪贴板。

2. 进入测试环境，粘贴命令，下载镜像。

3. 参考的启动容器的命令：

   ```
   docker run --gpus '"'device=0'"' -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -w /opt/nvidia/deepstream/deepstream-6.0 nvcr.io/nvidia/deepstream:6.0.1-devel
   ```

   或

   ```
   nvidia-docker run -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -w /opt/nvidia/deepstream/deepstream-6.0  nvcr.io/nvidia/deepstream:6.0.1-[CONTAINER-TAG]
   ```

   

4. 进入容器。

### 3. 启动rtsp服务器模拟输入源

本例中rtsp采用github上的实现:[aler9/rtsp-simple-server(github.com)](https://github.com/aler9/rtsp-simple-server)

步骤：

1. 下载对应系统版本的rtsp-simple-server压缩包倒测试环境并解压。
2. 根据需要调整rtsp-simple-server.yml配置文件。
3. 启动rtsp-simple-server.可采用nohup方式。

```
nohup ./rtsp-simple-server &
```

### 4. 使用ffmpeg将视频文件转换为rtsp源

参考命令：

```
ffmpeg -re -stream_loop -1 -i /opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_720p.mp4  -f rtsp -rtsp_transport tcp rtsp://localhost:18554/test
```

在容器/opt/nvidia/deepstream/deepstream-6.0/samples/streams目录下自带有一些测试视频。

注意这里的端口号和rtsp-simple-server.yml中配置的rtspAddress端口号保持一致。

### 5 启动demo

本文档中主要介绍python的demo。

python demo地址：[NVIDIA-AI-IOT/deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)

将代码下载到测试服务器容器中以下位置，demo中会用到一些容器中原有的资源。是以相对路径配置的。

```
cd /opt/nvidia/deepstream/deepstream-6.0/sources
git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git
```

demo路径:

```
/opt/nvidia/deepstream/deepstream-6.0/sources/deepstream_python_apps/apps
```

demo说明参见每个demo文件夹中的README文件。

对于纯控制台服务器，可选则deepstream-rtsp-in-rtsp-out测试，该例子接收rtsp数据源，识别完成后再转为rtsp输出。





