class DataDependentNet(nn.Module):
    def __init__(self):
        super(DataDependentNet, self).__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)
        self.linear3 = nn.Linear(5, 3)

    def forward(self, x):
        tmp = F.relu(self.linear1(x))
        # 有数据依赖的控制流：如果x的第一个元素大于0.5，使用linear2，否则使用linear3
        if tmp[0, 0] > 0.5:
            return self.linear2(tmp)
        else:
            return self.linear3(tmp)
