import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

# 支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

# 1. 类别名称（和训练一致）
class_names = [
    'cardboard (纸板)',
    'glass (玻璃)',
    'metal (金属)',
    'paper (纸张)',
    'plastic (塑料)',
    'trash (其他)'
]

# 2. 加载模型
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

# 3. 图片预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# 4. 读取并预测
img_path = 'test.jpg'
img = Image.open(img_path).convert('RGB')
tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    pred = probs.argmax().item()

print(f"预测结果：{class_names[pred]}")
print(f"置信度：{probs[pred]:.2%}")

# 5. 可视化
plt.figure(figsize=(5, 5))
plt.imshow(img)
plt.axis('off')
plt.title(f"预测：{class_names[pred]}\n置信度：{probs[pred]:.2%}", fontsize=14)
plt.tight_layout()
plt.savefig('result.jpg')
plt.show()