# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
# 解决matplotlib中文显示 
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用中文字体
plt.rcParams['axes.unicode_minus'] = False  

# 加入公开训练集
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 飞机(0)和汽车(1)
def filter_two_classes(images, labels, class1=0, class2=1):
    # 标签为1维数组
    labels = labels.flatten()
    # 创建掩码
    mask = (labels == class1) | (labels == class2)
    filtered_images = images[mask]
    filtered_labels = labels[mask]
    
    # 标签映射为0，1
    filtered_labels = np.where(filtered_labels == class1, 0, 1)
    return filtered_images, filtered_labels.reshape(-1, 1)

# 过滤数据集
train_images, train_labels = filter_two_classes(train_images, train_labels)
test_images, test_labels = filter_two_classes(test_images, test_labels)

# 预处理
def preprocess(data):
    data = data.astype('float32') / 255.0  # 歸一化
    return data

train_images2 = preprocess(train_images)
test_images2 = preprocess(test_images)

# 1像素值分布对比
plt.figure(figsize=(10, 6))
plt.hist(train_images.flatten(), bins=50, color='blue', alpha=0.5, label='原始像素')
plt.hist(train_images2.flatten(), bins=50, color='red', alpha=0.5, label='归一化像素')
plt.title('预处理前后像素值分布对比')
plt.xlabel('像素值')
plt.ylabel('频率')
plt.legend()
plt.savefig('pixel_distribution_comparison.png')
plt.show()

# 2前后图像对比
plt.figure(figsize=(15, 10))
plt.suptitle('预处理前后对比', fontsize=16)

class_names = ['飞机', '汽车']
# 随机选择5个样本
indices = np.random.choice(len(train_images), 5, replace=False)

for i, idx in enumerate(indices):
    # 预处理前
    plt.subplot(2, 5, i+1)
    plt.imshow(train_images[idx])
    plt.title(f"原始: {class_names[train_labels[idx][0]]}")
    plt.axis('off')
    
    # 预处理后
    plt.subplot(2, 5, i+6)
    # 注意：归一化后的图像需要调整显示范围
    plt.imshow(train_images2[idx])
    plt.title(f"归一化后")
    plt.axis('off')

plt.tight_layout()
plt.savefig('preprocessing_comparison.png')
plt.show()
# 例子
print(f"训练集: {train_images2.shape} {train_labels.shape}")
print(f"测试集: {test_images2.shape} {test_labels.shape}")
print(f"类别 - 飞机(0): {np.sum(train_labels == 0)}, 汽车(1): {np.sum(train_labels == 1)}")

def build_cnn_model(input_shape=(32, 32, 3), num_classes=2):  # 修改2输出
    """二分类的CNN"""
    model = models.Sequential([
        # 卷积层
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        
        # 全连接层
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')  # 输出层2输出
    ])
    return model

model = build_cnn_model()
model.summary()  # 确认结构

#训练参数配置
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
history = model.fit(
    train_images2, train_labels,
    epochs=20,  # 增加epoch數
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# 测试集评估
test_loss, test_acc = model.evaluate(test_images2, test_labels, verbose=2)
print(f"\n测试集准确率: {test_acc:.4f}")

# 可视化结果图像
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label="训练准确率")
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('准确率曲线 (飞机 vs 汽车)')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('损失曲线')
plt.legend()
plt.tight_layout()
plt.savefig('binary_classification_history.png')
plt.show()

# 预测
class_names = ['飞机', '汽车']
sample_images = test_images2[:5]
predictions = model.predict(sample_images)
predicted_classes = np.argmax(predictions, axis=1)

plt.figure(figsize=(15, 3))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(sample_images[i])
    plt.title(f"预测: {class_names[predicted_classes[i]]}")
    plt.axis('off')
plt.savefig('binary_predictions.png')
plt.show()
