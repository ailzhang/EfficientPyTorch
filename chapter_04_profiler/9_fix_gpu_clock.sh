# 查询
nvidia-smi --query-gpu=pstate,clocks.mem,clocks.sm,clocks.gr --format=csv

# clocks.current.memory [MHz], clocks.current.sm [MHz], clocks.current.graphics [MHz]
# 9751 MHz, 1695 MHz, 1695 MHz

# 查询GPU支持的clock组合
nvidia-smi --query-supported-clocks=gpu_name,mem,gr --format=csv

# 设置persistent mode
sudo nvidia-smi -pm 1

# 固定GPU时钟
nvidia-smi -ac 9751,1530 # <memory, graphics>
