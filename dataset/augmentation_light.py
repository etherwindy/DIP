from PIL import Image, ImageEnhance, ImageDraw
import random
import numpy as np
import os
import pandas as pd

# 随机亮度因子范围
range0 = (0.8, 1.25)
range1 = (1.25, 2)  # 可根据需要调整范围
range2 = (0.5, 0.8)  # 可根据需要调整范围

def enhance_images_in_folder(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹 '{folder_path}' 不存在。")
        return
    csv = pd.read_csv('label.csv')
    # 循环处理每个文件
    n = len(csv)
    for idx in range(n):
        image = csv.iloc[idx, 0]
        label = csv.iloc[idx, 1]
        image_path = os.path.join(folder_path, image)
        # 检查文件是否为图片
        if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # 打开图片文件
            try:
                flipped_img = change_light(image_path)
                new_image = "222"+image[3:]
                path = os.path.join(folder_path, new_image)
                # 保存增强后的图片
                flipped_img.save(path)
                data = {'image':new_image,'label':label}
                nd = pd.DataFrame(data,index=[0])
                nd.to_csv('label.csv',mode='a',header=False,index=False)
                print(f"已增强图片 '{image}' 并保存为 '{new_image}', 标签为{label}。")
            except Exception as e:
                print(f"处理文件 '{image}' 时出现错误：{e}")
        else:
            print(f"文件 '{image}' 不是图片文件，跳过翻转。")

def change_light(image_path):
    # 随机亮度因子范围
    range0 = (0.8, 1.25)
    range1 = (1.25, 2)  # 可根据需要调整范围
    range2 = (0.5, 0.8)  # 可根据需要调整范围

    # 随机生成亮度因子
    brightness_factor_lighter = random.uniform(*range1)
    brightness_factor_darker = random.uniform(*range2)


    img = Image.open(image_path)
    img_color = img.convert('RGB')

    # 获取图像数据并转换为NumPy数组
    img_array = np.array(img)

    avg_brightness = np.mean(img_array)

    # Adjust brightness based on the average brightness
    if avg_brightness > 80:  # Bright image
        brightness_factor = brightness_factor_darker
    elif avg_brightness < 40:  # Dark image
        brightness_factor = brightness_factor_lighter
    else:
        brightness_factor = random.uniform(*range0)

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Get image size and create a mask in the shape of a circle
    width, height = img.size
    mask = Image.new('L', (width, height), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse((0, 0, width, height), fill=255)

    # Enhance brightness within the circular mask
    enhancer = ImageEnhance.Brightness(img_color)
    img_color = enhancer.enhance(brightness_factor)


    # Create a black canvas
    black_canvas = Image.new('RGB', img.size, (0, 0, 0))

    # Paste the adjusted image onto the black canvas using the mask
    black_canvas.paste(img_color, mask=mask)
    return black_canvas

# 指定要翻转图片的文件夹路径和输出文件夹路径
folder_path = 'image'

# 调用函数翻转图片
enhance_images_in_folder(folder_path)