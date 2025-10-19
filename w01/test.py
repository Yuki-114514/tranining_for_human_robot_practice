import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("cuDNN enabled:", torch.backends.cudnn.enabled)
print("cuDNN version:", torch.backends.cudnn.version())

# 构造一个卷积运算（会调用 cuDNN）
x = torch.randn(10, 3, 224, 224).cuda()  # batch=10, 3通道, 224x224
conv = torch.nn.Conv2d(3, 32, 3).cuda()

y = conv(x)
print("Conv2D output shape:", y.shape)
