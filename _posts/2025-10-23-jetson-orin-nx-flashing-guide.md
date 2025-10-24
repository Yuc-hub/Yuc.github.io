---
title: "Jetson Orin NX  烧录 Ubuntu 20.04 和升级 Super 模式指南"
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

本教程将指导您如何在 Ubuntu 20.04 系统下为 Jetson Orin NX 刷机并启用 Super 模式。

## 准备工作

需要准备以下设备和环境：
- 一个用于 Jetson 的硬盘（需要格式化）
- 一台运行 Ubuntu 20.04 以上系统（amd64架构）的电脑或虚拟机
- SDKManager 或从官网下载的系统镜像

## 使用官方虚拟机和命令行方式刷机并升级 Super

### 1. 虚拟机准备

> 理论上任何 Ubuntu 20.04 及以上版本都可以，不一定要虚拟机。但使用虚拟机配置更简单。

虚拟机空间要求：**预留 70G**。可以从以下链接获取预配置的虚拟机镜像：

> https://www.yahboom.com/study/Jetson-Orin-NX 
> 提取密码：ntgy

![虚拟机下载页面](/assets/images/jetson-flashing-guide/image-20250701163820293.png)

输入提取码后刷新网站即可下载虚拟机镜像。

![下载虚拟机](/assets/images/jetson-flashing-guide/image-20250701163921159.png)

下载完成后解压，使用 VMware 打开。建议在虚拟机设置中分配 8GB 内存。

### 2. 安装 SDK Manager

虽然我们不使用 SDK Manager 进行刷机，但后续需要用它下载一些组件。官方虚拟机已预装，只需打开并登录 NVIDIA 账号，根据提示更新到最新版即可。

> 如需手动安装，可从官网下载：https://developer.nvidia.com/sdk-manager

安装步骤：

1. 在终端中进入下载目录，执行：
```bash
sudo dpkg -i sdkmanager_1.6.1-8175_amd64.deb  # 根据实际版本修改
```

2. 如果报告依赖问题，执行：
```bash
sudo apt --fix-broken install
```

3. 启动并登录：
- 终端运行 SDK Manager
- 点击 LOGIN，使用 NVIDIA 账号登录

![SDK Manager 登录界面](/assets/images/jetson-flashing-guide/500px-LOGIN水印-3.jpg)

### 3. 下载系统

从 [Jetpack 存档](https://developer.nvidia.com/embedded/jetpack-archive) 下载所需版本。**注意 Ubuntu 和 JetPack 版本对应关系**。本教程使用 Ubuntu 20.04 对应的 JetPack 5.1.5。

![JetPack 版本选择](/assets/images/jetson-flashing-guide/image-20250630135050222.png)

好消息是，JetPack 5.1.5 已支持在 Ubuntu 20.04 上启用 Super 模式，可以将功耗从 25W 提升至 40W，显著提升性能。

![性能提升说明](/assets/images/jetson-flashing-guide/image-20250701165410103.png)

选择版本后，在 Jetson Linux Page 页面下载这两个文件：

![下载必要文件](/assets/images/jetson-flashing-guide/image-20250701165635508.png)

下载后执行解压和准备命令：

```bash
tar xf ${L4T_RELEASE_PACKAGE}
sudo tar xpf ${SAMPLE_FS_PACKAGE} -C Linux_for_Tegra/rootfs/
cd Linux_for_Tegra/
sudo ./apply_binaries.sh
sudo ./tools/l4t_flash_prerequisites.sh
```

替换文件名示例：

![L4T Release Package](/assets/images/jetson-flashing-guide/image-20250701172219673.png)

![Sample FS Package](/assets/images/jetson-flashing-guide/image-20250701172234629.png)

> 详细步骤参考[官方教程](https://docs.nvidia.com/jetson/archives/r35.6.2/DeveloperGuide/IN/QuickStart.html)

**可能遇到的问题：**

1. `sudo ./apply_binaries.sh` 提示密钥过期
   - 可以忽略，通常不影响后续操作

2. `sudo ./tools/l4t_flash_prerequisites.sh` 提示缺少包
   - 按提示安装即可

3. 提示进程占用
   - 找到并结束相关进程

### 4. 进入 Recovery 模式

每个设备进入 Recovery 模式的方法可能不同：

- 官方 Nano/NX 开发套件：短接 Rec 引脚与 GND 引脚
- 达妙板卡：按以下步骤操作

推荐顺序：

1. **按住 REC 按钮，接通电源**
2. **保持 3 秒后松开 REC 按钮**
3. 用 USB 连接 Jetson 与主机（**注意只有特定的 Type-C 接口可用**）

Recovery 模式下风扇不会转动：

![Recovery 模式示意](/assets/images/jetson-flashing-guide/image-20250701171326012.png)

### 5. 系统烧录

找到 NVIDIA 文件夹中的 super 配置文件：
`nvidia_sdk/JetPack_5.1.5_Linux_JETSON_ORIN_NX_TARGETS/Linux_for_Tegra`

Orin NX 和 Orin Nano 使用相同配置：

![Super 配置文件](/assets/images/jetson-flashing-guide/image-20250630153258309.png)

使用以下命令进行烧录：

```bash
sudo ./tools/kernel_flash/l4t_initrd_flash.sh --external-device nvme0n1p1 \
  -c tools/kernel_flash/flash_l4t_external.xml -p "-c bootloader/t186ref/cfg/flash_t234_qspi.xml" \
  --showlogs --network usb0 --erase-all jetson-orin-nano-devkit-super internal
```

![烧录选项说明](/assets/images/jetson-flashing-guide/image-20250701172428566.png)

烧录过程中注意及时连接设备，避免超时：

![等待连接设备](/assets/images/jetson-flashing-guide/image-20250120155459747.png)

烧录完成后风扇会开始转动。断电重启即可进入系统：

![烧录完成](/assets/images/jetson-flashing-guide/image-20250120155405246.png)

**常见问题：**
- 卡在 "sending bct"
  - 通常是连接问题或未正确进入 Recovery 模式
  - 按上述步骤重试几次

### 6. 安装其他 SDK

在安装其他 SDK 前，需要先进行基本设置并启用 SSH：

```bash
# 安装必要包
sudo apt-get install sshpass
sudo apt-get install abootimg
sudo apt-get install nfs-kernel-server

# 检查 SSH 状态
sudo systemctl status ssh
```

如显示 "Active: active (running)"，则 SSH 服务已启动。

然后：

1. Jetson 关机断电
2. 主机打开 SDK Manager
3. 使 Jetson 进入 Recovery 模式

SDK Manager 会识别到设备：

![SDK Manager 识别设备](/assets/images/jetson-flashing-guide/image-20250709133345299.png)

注意事项：
- 取消勾选 "Host Machine"（避免下载不必要的 CUDA 环境）
- Target Hardware 可自动匹配（注意选择正确的内存版本）
- 选择之前使用的 JetPack 版本

![SDK 配置选择](/assets/images/jetson-flashing-guide/image-20250709133531253.png)

确认状态为 Recovery 模式后继续：

![确认 Recovery 状态](/assets/images/jetson-flashing-guide/image-20250701173815250.png)

因为已经手动安装了系统，取消勾选 "Jetson Linux"，同意协议后点击 "CONTINUE"。

到以下界面时：
1. 让 Jetson 重启进入系统（不是 Recovery 模式）
2. 输入系统账户名和密码
3. IP Address 保持默认 (192.168.55.1)

![SSH 配置](/assets/images/jetson-flashing-guide/image-20250701174019209.png)

等待安装完成。注意监控设备温度，过热可能导致失败。