# 快速开始指南

本文档帮助你快速运行人脸识别计算划分实验。

## 项目结构

```
OnDevice-Face-Recognition-Android/
├── app/                          # Android 应用源码
│   └── src/main/java/.../
│       ├── domain/
│       │   ├── embeddings/       # FaceNet 模型
│       │   ├── face_detection/   # 人脸检测和活体检测
│       │   └── offloading/       # 计算卸载模块 ⭐ (新增)
│       │       ├── OffloadingConfig.kt      # 配置
│       │       ├── CloudService.kt          # 云端通信
│       │       ├── OffloadingImageVectorUseCase.kt  # 卸载逻辑
│       │       └── PerformanceLogger.kt     # 性能日志
│       └── ...
├── server/                       # 云端服务器 ⭐ (新增)
│   ├── main.py                   # 服务器主程序
│   ├── benchmark.py              # 基准测试脚本
│   ├── requirements.txt          # Python 依赖
│   ├── start_server.sh           # 启动脚本
│   └── README.md                 # 服务器文档
├── docs/                         # 文档 ⭐ (新增)
│   ├── MODULE_ARCHITECTURE.md    # 模块架构分析
│   ├── EXPERIMENT_DESIGN.md      # 实验设计
│   ├── TEST_REPORT_TEMPLATE.md   # 测试报告模板
│   ├── DEMO_CHECKLIST.md         # 演示检查清单
│   └── QUICK_START.md            # 本文档
└── ...
```

## 快速启动

### Step 1: 启动云端服务器

```bash
cd server
./start_server.sh
```

或者手动启动：

```bash
cd server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p models
cp ../app/src/main/assets/facenet_512.tflite models/
uvicorn main:app --host 0.0.0.0 --port 8000
```

服务器启动后，访问 http://localhost:8000/docs 查看 API 文档。

### Step 2: 配置 Android 客户端

编辑 `app/src/main/java/.../domain/offloading/OffloadingConfig.kt`：

```kotlin
object OffloadingConfig {
    // 修改为你的服务器 IP
    var serverHost: String = "192.168.1.xxx"  
    var serverPort: Int = 8000
    
    // 设置当前模式
    var currentMode: OffloadingMode = OffloadingMode.LOCAL_ONLY
}
```

### Step 3: 添加网络权限

确保 `AndroidManifest.xml` 包含：

```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
```

### Step 4: 构建并运行应用

```bash
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### Step 5: 执行性能测试

在应用中切换不同模式，观察性能指标变化。

## 计算划分模式说明

| 模式 | 代码 | 本地执行 | 云端执行 | 传输数据 |
|------|------|---------|---------|---------|
| Mode 0 | `LOCAL_ONLY` | 全部 | 无 | 0 |
| Mode 1 | `EMBEDDING_OFFLOAD` | 检测+搜索 | 嵌入 | ~50KB |
| Mode 2 | `SEARCH_OFFLOAD` | 检测+嵌入 | 搜索 | ~2KB |
| Mode 3 | `EMBEDDING_AND_SEARCH_OFFLOAD` | 检测 | 嵌入+搜索 | ~50KB |
| Mode 4 | `FULL_OFFLOAD` | 无 | 全部 | ~200KB |

## 性能基准测试

运行服务器端基准测试：

```bash
cd server
python benchmark.py --server http://localhost:8000 --iterations 100
```

## 生成测试报告

1. 在应用中测试各种模式
2. 使用 `PerformanceLogger` 导出 CSV 数据
3. 根据 `docs/TEST_REPORT_TEMPLATE.md` 填写报告

## 演示准备

查看 `docs/DEMO_CHECKLIST.md` 获取详细的演示准备清单。

## 常见问题

### Q: 服务器无法连接？

1. 确认服务器正在运行
2. 检查防火墙设置
3. 确认手机和电脑在同一网络
4. 检查 IP 地址配置

### Q: 性能指标异常？

1. 确保网络稳定
2. 等待模型预热（前几次推理较慢）
3. 检查设备是否过热降频

### Q: 如何添加测试人脸？

在应用中使用 "Add Face" 功能添加人脸照片。

## 更多文档

- [模块架构分析](MODULE_ARCHITECTURE.md)
- [实验设计](EXPERIMENT_DESIGN.md)
- [测试报告模板](TEST_REPORT_TEMPLATE.md)
- [演示检查清单](DEMO_CHECKLIST.md)

