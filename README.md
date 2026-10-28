7万+数据集！无畏契约VaLoRant YOLO模型专用数据集
截图大小：256
截图数量：7w+
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d71a761d26bc40fb8ee499c960c40457.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cf4f321ebe7441e894129bdd09f4913c.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ba5a8c7e84834e53adbdb6f48a63ba5d.png)
：

✅ **7万+ 高质量截图**（256×256）  
✅ **标注类别**：玩家角色、敌人、血条、技能图标、枪械、地图标记等  
✅ **YOLOv8/v11 训练代码**（支持自定义配置）  
✅ **数据增强策略**（提升小目标检测能力）  
✅ **模型导出与推理脚本**

---

## ✅ 一、数据集概览

| 项目 | 内容 |
|------|------|
| **游戏** | 《Valorant》（无畏契约） |
| **截图数量** | 70,000+ 张 |
| **分辨率** | 256×256（统一裁剪） |
| **标注格式** | YOLO 格式（`.txt`，每张图对应一个标签文件） |
| **类别数** | 8 类（可扩展） |
| **标注工具** | LabelImg / Roboflow / CVAT |

### 🎯 标注类别（建议）

```yaml
names:
  - player          # 自己角色
  - enemy           # 敌方角色
  - teammate        # 队友
  - health_bar      # 血条（含护盾）
  - weapon          # 当前武器（Vandal、Phantom 等）
  - ability_icon    # 技能图标（Q/E/X/C）
  - minimap_marker  # 小地图标记
  - bomb_site       # 爆破点（A/B site）
```

> 💡 **说明**：血条和技能图标为**小目标**，需特别处理（如 mosaic 增强、高分辨率训练）。

---

## ✅ 二、目录结构

```
valorant_yolo_dataset/
├── images/
│   ├── train/       # 56,000 张
│   ├── val/         # 7,000 张
│   └── test/        # 7,000 张
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── dataset.yaml     # YOLO 数据集配置
└── README.md
```

---

## ✅ 三、`dataset.yaml` 配置文件

```yaml
# dataset.yaml
path: /path/to/valorant_yolo_dataset
train: images/train
val: images/val
test: images/test

nc: 8
names: ['player', 'enemy', 'teammate', 'health_bar', 'weapon', 'ability_icon', 'minimap_marker', 'bomb_site']
```

---

## ✅ 四、训练代码（YOLOv8 / YOLOv11）

### 1. 安装依赖

```bash
pip install ultralytics opencv-python numpy matplotlib
# 若使用 YOLOv11（假设为 Ultralytics 最新版）
# pip install git+https://github.com/ultralytics/ultralytics.git
```

### 2. 训练脚本 `train.py`

```python
# train.py
from ultralytics import YOLO

# 加载预训练模型（YOLOv8s 或 YOLOv11s）
model = YOLO('yolov8s.pt')  # 或 'yolov11s.pt'

# 训练配置
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=256,              # 与截图尺寸一致
    batch=64,               # 根据 GPU 显存调整
    workers=8,
    name='valorant_yolo_v8s',
    cache=True,             # 缓存图像加速训练
    augment=True,           # 启用默认增强
    hsv_h=0.015,            # 色调扰动
    hsv_s=0.7,              # 饱和度
    hsv_v=0.4,              # 亮度
    degrees=10.0,           # 旋转
    translate=0.1,          # 平移
    scale=0.5,              # 缩放（对小目标有效）
    mosaic=1.0,             # 强烈建议开启 mosaic（提升小目标检出）
    flipud=0.0,
    fliplr=0.5,
    patience=20,            # 早停
    device=0                # GPU ID
)

# 导出为 ONNX（用于部署）
model.export(format='onnx')
```

---

## ✅ 五、数据增强建议（针对小目标）

由于 **血条、技能图标** 在 256×256 图中可能仅占 10×5 像素，建议：

1. **开启 Mosaic 增强**（默认已启用）：将 4 张图拼接，增加小目标上下文。
2. **关闭 Copy-Paste**（避免标签错位）。
3. **使用更高输入尺寸训练**（如 512×512），推理时再缩放到 256。
4. **添加自定义增强**（如随机锐化、对比度拉伸）。

```python
# 自定义增强示例（在训练前预处理）
import cv2
import numpy as np

def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)
```

---

## ✅ 六、推理脚本（检测 Valorant 截图）

```python
# infer.py
from ultralytics import YOLO
import cv2

model = YOLO('runs/detect/valorant_yolo_v8s/weights/best.onnx')

# 单图推理
results = model('test.jpg')

# 可视化结果
annotated_img = results[0].plot()
cv2.imshow('Result', annotated_img)
cv2.waitKey(0)

# 批量推理
results = model(['img1.jpg', 'img2.jpg'], conf=0.4, iou=0.5)
for r in results:
    print(r.boxes.cls)  # 类别
    print(r.boxes.conf) # 置信度
```

---

## ✅ 七、性能优化建议

| 问题 | 解决方案 |
|------|--------|
| 小目标漏检 | 开启 `mosaic=1.0`，使用 `yolov8m` 或 `yolov8l` |
| 血条误检 | 添加负样本（无血条区域）到训练集 |
| 实时性要求高 | 导出 TensorRT 引擎（`model.export(format='engine')`） |
| 多类别混淆 | 检查标注一致性，使用 `cls_loss` 权重调整 |

---

## ✅ 八、部署到 Web 系统（可选）

结合您之前的系统架构，可将此模型集成到：

- **前端**：Vue3 上传 Valorant 截图
- **后端**：Flask 调用 ONNX 模型
- **输出**：返回 JSON + 标注图 + AI 分析（如“敌方位置分布”）

---


