# -*- coding: utf-8 -*-
"""
基于 CIFAR-10 的 CNN 二分类图像识别（飞机 vs 汽车）
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. 配置与初始化
# ============================================================

# 解决 matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子，保证实验可复现
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# 2. 加载与过滤数据集
# ============================================================

# 加载 CIFAR-10 公开数据集
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


def filter_two_classes(images, labels, class1=0, class2=1):
    """
    从数据集中筛选指定的两个类别，并将标签重新映射为 0 和 1
    
    参数:
        images: 图像数据
        labels: 标签数据
        class1: 目标类别1的原始编号（默认0=飞机）
        class2: 目标类别2的原始编号（默认1=汽车）
    
    返回:
        filtered_images: 筛选后的图像
        filtered_labels: 重新映射后的标签(0和1)
    """
    # 将标签展平为一维数组
    labels = labels.flatten()
    
    # 创建布尔掩码，筛选目标类别
    mask = (labels == class1) | (labels == class2)
    filtered_images = images[mask]
    filtered_labels = labels[mask]
    
    # 将标签重新映射为 0 和 1（方便二分类）
    filtered_labels = np.where(filtered_labels == class1, 0, 1)
    
    return filtered_images, filtered_labels.reshape(-1, 1)


# 过滤训练集和测试集（只保留飞机和汽车两类）
train_images, train_labels = filter_two_classes(train_images, train_labels)
test_images, test_labels = filter_two_classes(test_images, test_labels)

# ============================================================
# 3. 数据预处理（归一化）
# ============================================================


def preprocess(data):
    """像素值归一化：从 0~255 缩放到 0~1"""
    data = data.astype('float32') / 255.0
    return data


train_images_norm = preprocess(train_images)
test_images_norm = preprocess(test_images)

# ============================================================
# 4. 数据可视化
# ============================================================

# --- 4.1 像素值分布对比 ---
plt.figure(figsize=(10, 6))
plt.hist(train_images.flatten(), bins=50, color='blue', alpha=0.5, label='原始像素')
plt.hist(train_images_norm.flatten(), bins=50, color='red', alpha=0.5, label='归一化像素')
plt.title('预处理前后像素值分布对比')
plt.xlabel('像素值')
plt.ylabel('频率')
plt.legend()
plt.savefig('pixel_distribution_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 4.2 预处理前后图像对比 ---
plt.figure(figsize=(15, 10))
plt.suptitle('预处理前后对比', fontsize=16)
class_names = ['飞机', '汽车']

# 随机选择 5 个样本
indices = np.random.choice(len(train_images), 5, replace=False)

for i, idx in enumerate(indices):
    # 上排：原始图像
    plt.subplot(2, 5, i + 1)
    plt.imshow(train_images[idx])
    plt.title(f"原始: {class_names[train_labels[idx][0]]}")
    plt.axis('off')
    
    # 下排：归一化后图像
    plt.subplot(2, 5, i + 6)
    plt.imshow(train_images_norm[idx])
    plt.title(f"归一化后")
    plt.axis('off')

plt.tight_layout()
plt.savefig('preprocessing_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 4.3 打印数据集信息 ---
print("=" * 50)
print("数据集信息")
print("=" * 50)
print(f"训练集形状: {train_images_norm.shape}")
print(f"训练标签形状: {train_labels.shape}")
print(f"测试集形状: {test_images_norm.shape}")
print(f"测试标签形状: {test_labels.shape}")
print(f"训练集 - 飞机(0): {np.sum(train_labels == 0)} 张, 汽车(1): {np.sum(train_labels == 1)} 张")
print(f"测试集 - 飞机(0): {np.sum(test_labels == 0)} 张, 汽车(1): {np.sum(test_labels == 1)} 张")
print("=" * 50)

# ============================================================
# 5. 构建 CNN 模型
# ============================================================


def build_cnn_model(input_shape=(32, 32, 3), num_classes=2):
    """
    构建二分类卷积神经网络模型
    
    网络结构：
    Conv(32) → MaxPool → Conv(64) → MaxPool → Conv(128) → Flatten → Dense(512) → Dropout → Dense(2)
    """
    model = models.Sequential([
        # 第一组：卷积 + 池化
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # 第二组：卷积 + 池化
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # 第三组：卷积（不池化）
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        
        # 全连接分类器
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


# 创建模型
model = build_cnn_model()

# 打印模型结构
print("\n模型结构：")
model.summary()

# ============================================================
# 6. 编译与训练
# ============================================================

# 编译模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n开始训练...")

# 训练模型
history = model.fit(
    train_images_norm,
    train_labels,
    epochs=20,
    batch_size=32,
    validation_split=0.2,  # 20% 作为验证集
    verbose=1  # 显示训练进度条
)

# ============================================================
# 7. 测试集评估
# ============================================================

test_loss, test_acc = model.evaluate(test_images_norm, test_labels, verbose=2)
print(f"\n测试集准确率: {test_acc:.4f}")
print(f"测试集损失: {test_loss:.4f}")

# ============================================================
# 8. 训练过程可视化
# ============================================================

plt.figure(figsize=(12, 4))

# 左图：准确率曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="训练准确率", linewidth=2)
plt.plot(history.history['val_accuracy'], label='验证准确率', linewidth=2)
plt.title('准确率曲线（飞机 vs 汽车）')
plt.xlabel('训练轮数')
plt.ylabel('准确率')
plt.legend()
plt.grid(True, alpha=0.3)

# 右图：损失曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失', linewidth=2)
plt.plot(history.history['val_loss'], label='验证损失', linewidth=2)
plt.title('损失曲线')
plt.xlabel('训练轮数')
plt.ylabel('损失值')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 9. 预测结果可视化（含真实标签对比）
# ============================================================

class_names = ['飞机', '汽车']

# 取测试集前 10 张图片做预测
num_samples = 10
sample_images = test_images_norm[:num_samples]
sample_labels = test_labels[:num_samples].flatten()

# 模型预测
predictions = model.predict(sample_images)
predicted_classes = np.argmax(predictions, axis=1)
predicted_probs = np.max(predictions, axis=1)

# 绘制预测结果
plt.figure(figsize=(18, 4))

for i in range(num_samples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images[i])
    
    # 判断预测是否正确
    is_correct = predicted_classes[i] == sample_labels[i]
    color = 'green' if is_correct else 'red'
    status = '✓' if is_correct else '✗'
    
    # 标题显示：预测结果 + 置信度 + 正确/错误标记
    plt.title(
        f"预测: {class_names[predicted_classes[i]]}\n"
        f"真实: {class_names[sample_labels[i]]} {status}\n"
        f"置信度: {predicted_probs[i]:.1%}",
        color=color,
        fontsize=10
    )
    plt.axis('off')

plt.tight_layout()
plt.savefig('prediction_results.png', dpi=150, bbox_inches='tight')
plt.show()
