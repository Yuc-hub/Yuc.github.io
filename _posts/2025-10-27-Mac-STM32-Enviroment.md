---
title: "Mac 配置STM32嵌入式环境：STM32CubeMX + armgcc + Clion + openocd"
date: 2025-10-27
categories:
  - 教程
tags:
  - 嵌入式
  - MacOS
  - DSTM32
toc: true
toc_label: "目录"
toc_sticky: true
header:
  teaser: /assets/images/Mac-STM32-Enviroment/clion-banner.jpg
---


一直以来都是使用 windows 下的 keil5 进行 STM32 的开发，但是最近想尝试使用 Mac 进行 STM32 开发，奈何 keil5 没有 MacOS 版本，ARM 官方似乎也并没有想适配 MacOS 的计划。但是时代也确实变了，微软旗下 VSCode 和 JetBrain 旗下 CLion 都是很优秀的IDE，而且还内置 AI 助手和丰富的拓展插件，ARM 官方也似乎将重心放在了 STM32CubeMX 上，也推出了适用于STM32 开发的工具链和 VSCode 插件。所以正好在这个契机下想尝试适应更加现代化的 STM32 嵌入式开发环境，所以有了这个环境配置的记录。

本文录在macOS系统下配置STM32嵌入式开发环境的完整流程，包含以下核心组件：

- **STM32CubeMX** - 用于配置和生成芯片的基础代码
- **Clion** - 作为IDE进行开发
- **armgcc** - 作为编译工具链
- **openocd** - 用于烧录和调试

## 环境准备

### 1. 下载必要软件

#### STM32CubeMX
- 官网下载地址：[https://www.st.com/content/st_com/en/stm32cubemx.html#st-get-software](https://www.st.com/content/st_com/en/stm32cubemx.html#st-get-software)
- 选择macOS版本下载并安装

#### Clion
- 官网下载地址：[https://www.jetbrains.com/clion/](https://www.jetbrains.com/clion/)
- 选择macOS版本下载
- 注意：中国用户建议访问中文官网 [www.jetbrains.com.cn](https://www.jetbrains.com.cn) 获得更好的访问体验

#### Arm GNU Toolchain (armgcc)
- 下载地址：[Arm Developer官网](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads)
- 推荐下载最新稳定版本（当前为14.3.Rel1）
- 选择适合macOS的版本：
  - Apple Silicon芯片：`arm-gnu-toolchain-14.3.rel1-darwin-arm64-arm-none-eabi.pkg`
  - Intel芯片：选择对应x86_64版本

#### OpenOCD
`
bash
brew install openocd
`
### 2. 创建开发环境目录

建议新建一个专门的开发环境文件夹，例如：
`
bash
mkdir ~/DevEnv
cd ~/DevEnv
`

将下载的armgcc工具链解压到此目录中，保持环境整洁。

## 配置步骤

### 步骤1：使用STM32CubeMX创建项目

1. 打开STM32CubeMX，选择芯片型号
2. 配置引脚、时钟等系统设置
3. 切换到"Project Manager"选项卡：
   - 填写项目名称
   - 选择项目路径（**路径中不要包含中文字符**）
   - 在"Toolchain/IDE"中选择"CMake"
4. 点击"Generate Code"生成基础代码

### 步骤2：Clion项目配置

1. 使用Clion打开STM32CubeMX生成的项目文件夹
2. 首次打开时会弹出工具链配置窗口：
   - 系统会自动检测armgcc工具链，保持默认设置即可
   - 调试器选择"Bundled GDB"
3. 在运行配置中选择"debug-debug"
![image-20251027145447820]({{ "/assets/images/Mac-STM32-Enviroment/image-20251027145447820.png
" | relative_url }})
### 步骤3：烧录配置（OpenOCD）

1. 在Clion中创建新的OpenOCD烧录配置
2. 配置文件选择适合你开发板和调试器的配置：
   - - 嵌入式开发设置配置openocd和CLT
![image-20251027145529246]({{ "/assets/images/Mac-STM32-Enviroment/image-20251027145529246.png" | relative_url }})
   - 对于STM32F1系列和ST-Link调试器，使用`stm32f1_stlink.cfg`
![image-20251027145409292]({{ "/assets/images/Mac-STM32-Enviroment/image-20251027145409292.png" | relative_url }})
   - 配置文件内容示例：
`
source [find interface/stlink.cfg]
source [find target/stm32f1x.cfg]
reset_config none
`

## 验证安装

完成以上配置后，可以：

1. 在Clion中尝试编译项目
2. 连接开发板，测试烧录功能
3. 运行简单的LED闪烁程序验证环境是否正常工作

## 注意事项

- 确保所有路径不包含中文字符
- 定期更新工具链到最新版本以获得更好的兼容性
- 不同系列的STM32芯片需要选择对应的OpenOCD配置文件
- 如遇权限问题，可能需要给USB设备添加相应权限