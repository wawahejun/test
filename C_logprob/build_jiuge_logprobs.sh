#!/bin/bash
# -*- coding: utf-8 -*-

# Jiuge 模型 logprobs 功能编译脚本
# 根据 unified_api_plan.md 中的方案，编译支持 logprobs 的 InfiniCore-Infer 库

set -e  # 遇到错误立即退出

echo "=== Jiuge 模型 logprobs 功能编译脚本 ==="
echo "开始编译支持 logprobs 的 InfiniCore-Infer 库..."
echo

# 设置环境变量 - 指向安装目录而不是源码目录
export INFINI_ROOT="/home/wawahejun/.infini"
echo "INFINI_ROOT: $INFINI_ROOT"

# InfiniCore-Infer 目录
INFINI_INFER_ROOT="/home/wawahejun/reasoning/c_reasoning/InfiniCore-Infer"
echo "INFINI_INFER_ROOT: $INFINI_INFER_ROOT"

# 设置库路径
export LD_LIBRARY_PATH="/home/wawahejun/.infini/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="/home/wawahejun/.infini/lib/pkgconfig:$PKG_CONFIG_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# 检查必要的目录和文件
if [ ! -d "$INFINI_ROOT" ]; then
    echo "❌ 错误: InfiniCore 目录不存在: $INFINI_ROOT"
    exit 1
fi

if [ ! -d "$INFINI_INFER_ROOT" ]; then
    echo "❌ 错误: InfiniCore-Infer 目录不存在: $INFINI_INFER_ROOT"
    exit 1
fi

if [ ! -f "$INFINI_INFER_ROOT/include/infinicore_infer/models/jiuge.h" ]; then
    echo "❌ 错误: jiuge.h 头文件不存在"
    exit 1
fi

# 检查源文件是否存在
if [ ! -f "$INFINI_INFER_ROOT/src/models/jiuge/jiuge.cpp" ]; then
    echo "❌ 错误: jiuge.cpp 源文件不存在"
    exit 1
fi

echo "✅ 必要文件检查通过"
echo

# 进入 InfiniCore-Infer 目录
cd "$INFINI_INFER_ROOT"
echo "当前目录: $(pwd)"

# 检查是否存在构建系统
if [ -f "xmake.lua" ]; then
    echo "发现 xmake 构建系统"
    BUILD_SYSTEM="xmake"
elif [ -f "CMakeLists.txt" ]; then
    echo "发现 CMake 构建系统"
    BUILD_SYSTEM="cmake"
elif [ -f "Makefile" ]; then
    echo "发现 Makefile 构建系统"
    BUILD_SYSTEM="make"
elif [ -f "build.sh" ]; then
    echo "发现自定义构建脚本"
    BUILD_SYSTEM="script"
else
    echo "❌ 错误: 未找到构建系统 (xmake.lua, CMakeLists.txt, Makefile, 或 build.sh)"
    exit 1
fi

echo "构建系统: $BUILD_SYSTEM"
echo

# 创建构建目录
BUILD_DIR="build_logprobs"
if [ -d "$BUILD_DIR" ]; then
    echo "清理现有构建目录: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
echo "创建构建目录: $BUILD_DIR"
echo

# 根据构建系统进行编译
case $BUILD_SYSTEM in
    "xmake")
        echo "=== 使用 xmake 构建 ==="
        
        echo "清理之前的构建..."
        xmake clean
        
        echo "配置 xmake..."
        xmake config --mode=release
        
        echo "开始编译..."
        xmake build
        
        echo "安装库文件..."
        xmake install
        
        echo "编译完成"
        ;;
        
    "cmake")
        echo "=== 使用 CMake 构建 ==="
        cd "$BUILD_DIR"
        
        echo "配置 CMake..."
        cmake .. -DCMAKE_BUILD_TYPE=Release
        
        echo "开始编译..."
        make -j$(nproc)
        
        echo "编译完成"
        ;;
        
    "make")
        echo "=== 使用 Makefile 构建 ==="
        echo "开始编译..."
        make clean || true
        make -j$(nproc)
        
        echo "编译完成"
        ;;
        
    "script")
        echo "=== 使用自定义构建脚本 ==="
        echo "运行构建脚本..."
        bash build.sh
        
        echo "构建脚本执行完成"
        ;;
esac

echo
echo "=== 验证编译结果 ==="

# 检查库文件是否生成
LIB_PATHS=(
    "lib/libinfinicore_infer.so"
    "$BUILD_DIR/lib/libinfinicore_infer.so"
    "$BUILD_DIR/libinfinicore_infer.so"
    "libinfinicore_infer.so"
)

LIB_FOUND=false
for lib_path in "${LIB_PATHS[@]}"; do
    if [ -f "$lib_path" ]; then
        echo "✅ 找到库文件: $lib_path"
        LIB_FOUND=true
        
        # 检查库文件中是否包含新的符号
        echo "检查库文件中的 logprobs 相关符号..."
        if nm -D "$lib_path" 2>/dev/null | grep -q "inferBatchWithLogprobs"; then
            echo "✅ 找到 inferBatchWithLogprobs 符号"
        else
            echo "⚠️  警告: 未找到 inferBatchWithLogprobs 符号"
        fi
        
        if objdump -T "$lib_path" 2>/dev/null | grep -q "inferBatchWithLogprobs"; then
            echo "✅ 确认 inferBatchWithLogprobs 符号已导出"
        else
            echo "⚠️  警告: inferBatchWithLogprobs 符号可能未正确导出"
        fi
        
        break
    fi
done

if [ "$LIB_FOUND" = false ]; then
    echo "❌ 错误: 未找到编译生成的库文件"
    echo "请检查编译过程是否有错误"
    exit 1
fi

echo
echo "=== 设置环境变量 ==="
echo "请在运行测试前设置以下环境变量:"
echo "export INFINI_ROOT=$INFINI_ROOT"
echo "export LD_LIBRARY_PATH=$INFINI_ROOT/lib:$INFINI_ROOT/$BUILD_DIR/lib:$INFINI_ROOT/$BUILD_DIR:\$LD_LIBRARY_PATH"
echo

echo "=== 编译完成 ==="
echo "✅ Jiuge 模型 logprobs 功能编译成功！"
echo "现在可以运行测试脚本: python /home/wawahejun/reasoning/test/test_jiuge_logprobs.py <模型路径>"
echo