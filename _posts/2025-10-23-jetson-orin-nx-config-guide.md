---
title: "Jetson Orin NX 开荒记录"
date: 2025-10-23
categories:
  - 教程
tags:
  - Jetson
  - NVIDIA
  - 嵌入式系统
  - 开发板配置
toc: true
toc_label: "目录"
toc_sticky: true
header:
  teaser: /assets/images/jetson-flashing-guide/fengmian.jpg
---

# 英伟达Jetson Orin NX学习总结

## 一、Jetson介绍

Jetson Orin NX是一款嵌入式 GPU计算平台，模组极其小巧，功率可在 10 瓦到 20 瓦之间进行配置。此模组的性能可高达 Jetson AGX Xavier 的 3 倍、Jetson Xavier NX 的 5 倍。Jetson Orin NX 有 16GB 和 8GB 两个版本，本文使用版本为8GB版本，算力为 70 TOPS,在更新JetPack6.2系统后可以完成SUPER模式下的性能提升，算力可提升至117TOPS。核心板如下图所示，套件除了核心板还有散热器和外壳等。

<img src="{{ '/assets/images/jetson-config-guide/image-20250219112638643.png' | relative_url }}" alt="image-20250219112638643" style="zoom:60%;" />

## 二、环境搭建

### 1.网络连接

(1) 点击右上角的状态栏，点击Settings,选择Wi-Fi,点击需要连接的网络：

<img src="{{ '/assets/images/jetson-config-guide/Screenshot2023-11-22 -5-09-30.png' | relative_url }}" alt="Screenshot from 2023-11-22 05-09-30" style="zoom:30%;" />

(2) 这里以西电校园网为例，点击弹出的窗口，进入校园网登录界面：

<img src="{{ '/assets/images/jetson-config-guide/Screenshot2025-02-23-18-10-55-1741245898679-4.png' | relative_url }}" alt="Screenshot from 2025-02-23 18-10-55" style="zoom:20%;" />

输入账号密码即可登录。

### 2.配置中文输入法

(1) 点击右上角状态栏，点击Settings:

<img src="{{ '/assets/images/jetson-config-guide/Screenshot19-46-45-1741245898679-15.png' | relative_url }}" alt="Screenshot from 2025-03-01 19-46-45" style="zoom:60%;" />

(2) 点击Region & Language,之后点击Manage Installed Languages:

<img src="{{ '/assets/images/jetson-config-guide/Screenshot19-47-07-1741245898679-5.png' | relative_url }}" alt="Screenshot from 2025-03-01 19-47-07" style="zoom:50%;" />

(3) 如果弹出下图中的弹窗，点击Install:

<img src="{{ '/assets/images/jetson-config-guide/image-20250302153242319-1741245898679-6.png' | relative_url }}" alt="image-20250302153242319" style="zoom:60%;" />

(4) 之后点击Install/Remove Languages,在弹出的窗口中选择Chinese(simplified),点击Apply$\rightarrow$Apply System-Wide$\rightarrow$Close.

<img src="{{ '/assets/images/jetson-config-guide/Screenshot19-48-57-1740901447043-6-1741245898679-7.png' | relative_url }}" alt="Screenshot from 2025-03-01 19-48-57" style="zoom:50%;" />

(5) 打开终端,输入以下指令安装中文输入法：

```bash
sudo apt update
sudo apt install ibus-pinyin -y
```

(6) 安装完成后重启系统，进入Settings$\rightarrow$Keyboard添加键盘输入源：

<img src="{{ '/assets/images/jetson-config-guide/Screenshot19-47-52-1741245898679-8.png' | relative_url }}" alt="Screenshot from 2025-03-01 19-47-52" style="zoom:50%;" />

(7) 点击三个圆点的选项，再点击Others,在列表中选择智能拼音输入法，点击Add即可：

![Screenshot from 2025-03-01 19-48-09]({{ '/assets/images/jetson-config-guide/Screenshot19-48-09-1741245898679-9.png' | relative_url }})

(8) 之后在右上角的状态栏即可看到成功添加中文输入法。如果看到有中文输入法选项但是仍无法输出中文，可以尝试重启系统。

![Screenshot from 2025-02-24 21-41-27]({{ '/assets/images/jetson-config-guide/Screenshot21-41-27-1741245898679-10.png' | relative_url }})

### 3.安装火狐浏览器

如果在一开始的系统安装时跳过了火狐浏览器的安装，那么Jetson的系统是没有浏览器可以使用的。此时如果我们想安装火狐浏览器，在终端输入以下指令即可：

```bash
sudo apt update
sudo apt install firefox
```

安装完成后，火狐浏览器的标志便会出现在左侧的应用栏里。

![Screenshot from 2025-02-23 18-01-08]({{ '/assets/images/jetson-config-guide/Screenshot18-01-08-1741245898679-11.png' | relative_url }})

### 4.切换下载源

在 Ubuntu 系统中，镜像源（软件源）是存储 Ubuntu 软件包的服务器，通过这些源可以下载和安装软件包以及系统更新。更换镜像源能提高 apt 相关命令的稳定性与速度。这里以更换清华大学镜像源为例:

(1) 首先进入清华大学ubuntu ports镜像源官网https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu-ports/，选择系统对应ubuntu版本,复制镜像源内容。

<img src="{{ '/assets/images/jetson-config-guide/image-20250228205312861-1741245898679-12.png' | relative_url }}" alt="image-20250228205312861" style="zoom:40%;" />

```bash
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-backports main restricted universe multiverse

# 以下安全更新软件源包含了官方源与镜像站配置，如有需要可自行修改注释切换
deb http://ports.ubuntu.com/ubuntu-ports/ jammy-security main restricted universe multiverse
# deb-src http://ports.ubuntu.com/ubuntu-ports/ jammy-security main restricted universe multiverse

# 预发布软件源，不建议启用
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-proposed main restricted universe multiverse
# # deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-proposed main restricted universe multiverse
```

(2) 返回Jetson桌面，打开终端，备份镜像源设置文件：

```bash
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
```

(3) 之后使用vim编辑镜像源设置文件:

```bash
sudo vim /etc/apt/sources.list
```

进入vim后按下键盘上的i或a进入编辑模式，删除原有内容，将新的镜像源粘贴进去。之后按下esc退出编辑模式，再按下```:wq```保存并退出。**注意先按下```:```后再输```wq```,不要忽略了```:```这个符号。*

(4) 最后更新apt 本地索引：

```bash
sudo apt-get update
```

### 5.安装Jtop

Jtop是 Jetson 开发环境中的一个轻量级、高效的系统监控工具，用于实时监测和分析设备的资源使用情况,主要围绕 NVIDIA Jetson 开发板所运行的 Linux 系统，通过直观的命令行界面（CLI）展示设备的硬件性能信息，包括 CPU、GPU、内存、温度等状态。下面我们来介绍如何安装Jtop。

(1) Jtop的安装需要Python3和pip3,因此我们先安装这两个环境：

```bash
sudo apt update
sudo apt install python3
sudo apt-get update
sudo apt-get install python3-pip
```

通过查看pip3版本的指令可以查看pip3是否安装成功:

```bash
pip3 --version
```

<img src="{{ '/assets/images/jetson-config-guide/Screenshot13-41-29-1741245898679-13.png' | relative_url }}" alt="Screenshot from 2025-02-22 13-41-29" style="zoom:50%;" />

(2) 输入以下指令安装Jtop:

```
sudo pip3 install -U jetson-stats
```

(3) 安装完成后输入```jtop```即可启动Jtop。

<img src="{{ '/assets/images/jetson-config-guide/Screenshot19-06-05-1741245898679-14.png' | relative_url }}" alt="Screenshot from 2025-02-24 19-06-05" style="zoom:50%;" />

### 6.增加虚拟内存swap空间

SWAP 是指在计算机的硬盘上预留一部分空间，用作虚拟内存，也称为交换空间。虚拟内存是一种扩展计算机物理内存（RAM）的机制，允许计算机在物理内存不足时，将部分数据暂时存储在硬盘上，以释放物理内存供其他程序使用。SWAP 空间可以帮助避免因为内存不足而导致系统崩溃或程序异常退出。在某些情况下，系统可能会通过将不活动的数据交换到 SWAP 来保持正常运行。

(1) 首先输入```jtop```指令查看当前swap空间大小。开发板物理内存为8GB时，swap空间最好设置为8GB。我的初始swap空间为3.7GB，所以需要增加4GB。依次输入以下指令：

```bash
#1）新增swapfile文件大小自定义
sudo fallocate -l 4G /var/swapfile
#2）配置该文件的权限
sudo chmod 600 /var/swapfile
#3）建立交换分区
sudo mkswap /var/swapfile
#4）启用交换分区
sudo swapon /var/swapfile
```

(2) 再次输入```jtop```查看扩容后的空间，可以发现已经拓展到7.7GB。

<img src="{{ '/assets/images/jetson-config-guide/Screenshot from 2025-03-01 19-45-12-1741245898679-16.png' | relative_url }}" alt="Screenshot from 2025-03-01 19-45-12" style="zoom:70%;" />

(3) 最后设置自启动swapfile，输入命令：

```bash
sudo bash -c 'echo "/var/swapfile swap swap defaults 0 0" >> /etc/fstab'
```

### 7.通过SDKManager安装必要环境

NVIDIA的SDK Manager为Jetson提供端到端的开发环境设置解决方案,其中一些环境是我们进行人工智能模型开发所必需的，比如CUDA等。接下来我们介绍如何下载及使用SDKManager。

 (1) 下载SDKManager的压缩包

![image-20251024194708995]({{ '/assets/images/jetson-config-guide/image-20251024194708995.png' | relative_url }})

首先在虚拟机系统中打开英伟达官网中SDKManager的网页https://developer.nvidia.com/sdk-manager，找到.deb文件的下载按钮，点击下载。

 (2) 安装SDKManager

<img src="{{ '/assets/images/jetson-config-guide/2-1741245898679-17.png' | relative_url }}" alt="2" style="zoom:40%;" />

找到下载好的文件，在当前路径右键选择进入终端，输入以下指令:

```bash
sudo apt install ./sdkmanager_[version]-[build#]_amd64.deb
# [version]-[build#]是需要我们自己根据下载的版本号进行替换的
# 比如说这里根据的我的版本，就是sudo apt install ./sdkmanager_2.2.0-12028_amd64.deb
```

有可能会出现如下报错：

<img src="{{ '/assets/images/jetson-config-guide/3-1741245898679-18.png' | relative_url }}" alt="3" style="zoom:50%;" />

不用理会，不影响后续正常安装。

之后执行以下四条指令：

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/[distro]/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install sdkmanager
#[distro]为虚拟机系统的版本号，支持的版本号为ubuntu1804, ubuntu2004, ubuntu2204，根据自己系统的版本号进行选择
```

 (3) 启动SDKManager

在当前终端输入：```sdkmanager```即可启动。

注意：使用虚拟机环境时如果没有全屏并且虚拟机内存分配过小时会提示如下警告：

<!-- 本地绝对路径图片：不可用于网站托管。请把此图片复制到仓库的 assets/images/jetson-config-guide/ 下并替换下面的路径 -->
<img src="{{ '/assets/images/jetson-config-guide/截图 2025-02-25 09-10-20-1741245898680-36.png' | relative_url }}" alt="截图 2025-02-25 09-10-20" style="zoom:67%;" />

此时只需全屏虚拟机即可，否则会出现显示问题。内存过小问题可以不用解决。进入后需要登录英伟达账号，之后即可进入到如下安装界面：

<img src="{{ '/assets/images/jetson-config-guide/4-1741245898679-20.png' | relative_url }}" alt="4" style="zoom:40%;" />

进入到安装界面后，需要将Jetson与虚拟机使用Type-C线进行连接，注意连接到虚拟机上，不要连到主机上！同时Jetson需要正常开机进入系统。连接正常后在第二个红框中会显示连接状态。勾选上面四个红框的内容，之后点击CONTINUE下一步：

<img src="{{ '/assets/images/jetson-config-guide/5-1741245898679-21.png' | relative_url }}" alt="5" style="zoom:40%;" />

在这一步中把Jetson Linux选项取消勾选，剩余选项全部勾选，点击CONTINUE，进入下一步：

<img src="{{ '/assets/images/jetson-config-guide/6-1741245898679-22.png' | relative_url }}" alt="6" style="zoom:40%;" />

这里输入虚拟机系统的密码。

<img src="{{ '/assets/images/jetson-config-guide/7-1741245898679-23.png' | relative_url }}" alt="7" style="zoom:40%;" />

输入Jetson设备的用户名和密码。用户名是安装系统时自己设置的用户名，不一定叫jetson,要看自己设置了什么用户名。其余默认不需要更改，点击Install。

<img src="{{ '/assets/images/jetson-config-guide/8-1741245898679-24.png' | relative_url }}" alt="8" style="zoom:40%;" />

此时便进入了安装过程，耐心等待安装完成。建议在网络好的情况下进行安装，避免出现网络错误。

<img src="{{ '/assets/images/jetson-config-guide/9-1741245898679-25.png' | relative_url }}" alt="9" style="zoom:40%;" />

此时安装完成，点击FINISH即可完成安装。安装过程中有可能会出现部分组件安装失败的情况，点击重新安装即可。安装完成后需要重启Jetson设备，之后更新便可以生效。在Jetson的终端输入```jtop```,查看更新后组件的信息：

<img src="{{ '/assets/images/jetson-config-guide/Screenshot19-45-25-1741245898679-26.png' | relative_url }}" alt="Screenshot from 2025-03-01 19-45-25" style="zoom:70%;" />

### 8.Pycharm安装

(1) 首先进入Pycharm官网https://www.jetbrains.com/pycharm/download/other.html，找到Pycharm社区版，**下载Linux ARM64版**。

<img src="{{ '/assets/images/jetson-config-guide/screanshot135758-1741245898679-27.png' | relative_url }}" alt="屏幕截图 2025-03-01 135758" style="zoom:40%;" />

(2) 下载完成后进入Downloads文件夹，进入终端，输入以下指令进行解压操作：

```bash
# pycharm-<version>更换为下载的pycharm版本名
tar -xzf pycharm-<version>.tar.gz
```

<img src="{{ '/assets/images/jetson-config-guide/Screenshot15-00-20-1741245898679-28.png' | relative_url }}" alt="Screenshot from 2025-02-24 15-00-20" style="zoom:30%;" />

(3) 将解压后的文件移动至/opt文件夹下，利用```cd```指令进入Pycharm文件夹中的bin文件夹，再利用```sh```指令启动Pycharm:

```bash
# 这里以我下载的版本为例
sudo mv pycharm-community-2024.3.3 /opt
cd /opt/pycharm-community-2024.3.3/bin
sh pycharm.sh
```

<img src="{{ '/assets/images/jetson-config-guide/Screenshot15-13-46-1741245898679-29.png' | relative_url }}" alt="Screenshot from 2025-02-24 15-13-46" style="zoom:40%;" />

可以看到Pycharm正常启动。

(4) 进入初始页面后，点击左下角的齿轮，点击Create Desktop Entry，即可在菜单中添加启动快捷方式:

<img src="{{ '/assets/images/jetson-config-guide/Screenshot15-13-57-1741245898680-30.png' | relative_url }}" alt="Screenshot from 2025-02-24 15-13-57" style="zoom:50%;" />

<img src="{{ '/assets/images/jetson-config-guide/Screenshot15-24-33-1741245898680-31.png' | relative_url }}" alt="Screenshot from 2025-02-24 15-24-33" style="zoom:30%;" />

(5) 启动Pycharm后，在左侧应用栏中右键Pycharm图标，点击Add to Favorites,即可将Pycharm图标固定至左侧应用栏，方便大家以后启动软件。

<img src="{{ '/assets/images/jetson-config-guide/Screenshot15-35-54-1741245898680-32.png' | relative_url }}" alt="Screenshot from 2025-02-24 15-35-54" style="zoom:50%;" />

### 9.安装Pytorch

PyTorch是一个用于机器学习和深度学习的开源深度学习框架，由Facebook于2016年发布，其主要实现了自动微分功能，并引入动态计算图使模型建立更加灵活。现如今，Pytorch已经成为开源机器学习系统中，在科研领域市场占有率最高的框架，其在AI顶会上的占比在2022年已达80％。

不同版本的JecPack需要安装对应版本的Pytorch。下面针对最新版的JetPack 6.2,在Pycharm中进行Pytorch的安装。

(1) 首先安装依赖环境。在系统终端执行以下指令:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y install libcusparselt0 libcusparselt-dev -y
sudo apt remove python3-sympy
```

(2) 打开Pycharm,新建一个项目。建议使用默认项目名PycharmProject,因为后面所有项目的虚拟环境将使用这个项目下的虚拟环境。

(3) 下载Pytorch安装包进行安装。下载网址为:https://developer.download.nvidia.cn/compute/redist/jp/v61/pytorch/.之后复制安装包所在的路径，在Pycharm的终端输入:

```
pip install /home/jetson/Downloads/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
# 后面部分的路径需要改成自己文件所在路径
```

<img src="{{ '/assets/images/jetson-config-guide/Screenshot19-01-31-1741245898680-33.png' | relative_url }}" alt="Screenshot from 2025-03-01 19-01-31" style="zoom:50%;" />

(4) 验证安装。安装完成后,在当前.py文件下输入以下代码:

```python
import torch
print(torch.cuda.is_available())
```

点击运行，查看输出为True。说明PyTorch已经成功安装。

<img src="{{ '/assets/images/jetson-config-guide/Screenshot19-02-59-1741245898680-37.png' | relative_url }}" alt="Screenshot from 2025-03-01 19-02-59" style="zoom:50%;" />

可以看到这里有Numpy的Warning信息。解决方法为：卸载当前版本的Numpy,安装老版本的Numpy。例如我这里重新安装了1.24.0版本的Numpy,之后便不再有Warning。在Pycharm终端执行以下指令:

```bat
pip uninstall numpy
pip install numpy==1.24.0
```

<img src="{{ '/assets/images/jetson-config-guide/Screenshot19-05-17-1741245898680-34.png' | relative_url }}" alt="Screenshot from 2025-03-01 19-05-17" style="zoom:67%;" />

### 10.安装Torchvision

Torchvision是一个用于计算机视觉任务的PyTorch软件包。它是构建在PyTorch深度学习框架之上的一个附加库,提供了许多用于处理图像和视频数据的工具和函数。

(1) 首先安装依赖环境。在系统终端执行以下指令:

```bat
sudo apt update
sudo apt install ninja-build libwebp-dev libjpeg-dev -y
```

(2) 下载Torchvision安装包进行安装。下载网址为:https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl.**注意Torchvision的版本也需要和PyTorch版本进行对应，这里下载的Torchvision只和上一小节下载的PyTorch版本对应**。

(3) 复制安装包所在的路径，在Pycharm的终端输入:

```
pip install /home/jetson/Downloads/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
# 后面部分的路径需要改成自己文件所在路径
```

<img src="{{ '/assets/images/jetson-config-guide/Screenshot19-10-54-1741245898680-35.png' | relative_url }}" alt="Screenshot from 2025-03-01 19-10-54" style="zoom:50%;" />

(4) 验证安装。安装完成后,在当前.py文件下输入以下代码:

```py
import torch
print(f'Torchvision: {torchvision.__version__}')
```

点击运行，查看输出为Torchvision的版本号。说明Torchvision已经成功安装。

<img src="{{ '/assets/images/jetson-config-guide/Screenshot19-12-12.png' | relative_url }}" alt="Screenshot19-12-12" style="zoom:50%;" />