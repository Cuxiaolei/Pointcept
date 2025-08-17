# configs/_base_/dataset/s3dis.py
# 三标签分类的类别信息
class_names = ["class0", "class1", "class2"]  # 替换为你的类别名称
num_classes = 3
ignore_index = -1  # 无无效标签则保持-1

# 基础数据配置（主配置会复用这些参数）
data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=class_names,
    # 数据加载器的通用参数
    loader=dict(
        batch_size=2,
        num_workers=4,
        shuffle=True,
    ),
)