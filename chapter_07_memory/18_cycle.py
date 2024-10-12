import torch


class CustomLayer(torch.nn.Module):
    def __init__(self, model):
        super(CustomLayer, self).__init__()
        self.model = model


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.custom_layer = CustomLayer(self)


model = MyModel()
