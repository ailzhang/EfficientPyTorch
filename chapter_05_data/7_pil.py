from PIL import Image
import time


def resize_image(image_path, output_size):
    with Image.open(image_path) as img:
        img = img.resize(output_size)
        img.save("output.png")


image_path = "example.png"
output_size = (4096, 4096)  # 新的尺寸

# 开始计时
start_time = time.time()

# 执行图像缩放
resize_image(image_path, output_size)

# 计算耗时
duration = time.time() - start_time
print(f"Time taken: {duration} seconds")
