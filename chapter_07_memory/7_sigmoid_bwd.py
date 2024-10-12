# Sigmoid算子
out = 1 / (1 + exp(-x))

# Sigmoid反向算子
dx = dout * out * (1 - out)
