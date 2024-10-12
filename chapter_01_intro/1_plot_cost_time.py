import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Heiti TC'  # 替换为你选择的字体

models = ["GPT-1.3B", "GPT-2.7B", "GPT-6.7B", "GPT-13B", "GPT-30B", "GPT-70B"]
costs = [2000, 6000, 30000, 100000, 450000, 2500000]
training_days = [0.14, 0.48, 2.32, 7.43, 35.98, 176.55]
model_indices = np.arange(len(models))


# Plotting the bar chart with cost and training time using a more optimized approach with Chinese labels and units for training days
fig, ax1 = plt.subplots(figsize=(12, 6))

# Creating the first bar chart for costs
bars1 = ax1.bar(model_indices, costs, width=0.4, label='训练成本 (美元)', color='skyblue', align='center')
ax1.set_xlabel('模型')
ax1.set_ylabel('训练成本 (美元)')
ax1.set_yscale('log')
ax1.set_title('GPT系列模型训练成本和时间')
ax1.set_xticks(model_indices)
ax1.set_xticklabels(models)

# Adding text on top of the cost bars with shortened numbers
for bar in bars1:
    height = bar.get_height()
    if height >= 1e6:
        label = f'${height/1e6:.1f}M'
    elif height >= 1e3:
        label = f'${height/1e3:.1f}k'
    else:
        label = f'${height:.0f}'
    ax1.text(bar.get_x() + bar.get_width() / 2.0, height, label, ha='center', va='bottom')

# Creating the second bar chart for training days on the same x-axis
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
bars2 = ax2.bar(model_indices + 0.4, training_days, width=0.4, label='训练时间 (天)', color='lightcoral', align='center')
ax2.set_yscale('log')
ax2.set_ylabel('训练时间 (天)')

# Adding text on top of the training days bars with "天" unit
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f} 天', ha='center', va='bottom')

# Adding legends
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes)

plt.show()
