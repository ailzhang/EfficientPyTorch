# 安装
sudo apt install cpufrequtils

# 设置最大/最小频率
sudo cpufreq-set -r -g performance
sudo cpufreq-set -r -d 2Ghz
sudo cpufreq-set -r -u 2Ghz
