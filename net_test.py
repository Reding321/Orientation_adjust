from torch import nn
import torch


net = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(64 * 16 * 16, 64), nn.Sigmoid(),
    nn.Linear(64, 8)
)

pthfile = r"C:\Users\HP\Desktop\pythonProject\Orientation_adjust_tool_Remake\data\Ori_C0.pth"
net.load_state_dict(torch.load(pthfile, map_location=torch.device('cpu')))

X = torch.rand(size=(3, 1, 256, 256))
print(net(X))


