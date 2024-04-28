from PIL import Image
import pandas as pd
import os

def flip_images_in_folder(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹 '{folder_path}' 不存在。")
        return


    csv = pd.read_csv('label_test.csv')
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
                img = Image.open(image_path)
                # 翻转图片
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                new_image = "111"+image[3:]
                path = os.path.join(folder_path, new_image)
                # 保存翻转后的图片
                flipped_img.save(path)
                data = {'image':new_image,'label':label}
                nd = pd.DataFrame(data,index=[0])
                nd.to_csv('label_test.csv',mode='a',header=False,index=False)
                print(f"已翻转图片 '{image}' 并保存为 '{new_image}', 标签为{label}。")
            except Exception as e:
                print(f"处理文件 '{image}' 时出现错误：{e}")
        else:
            print(f"文件 '{image}' 不是图片文件，跳过翻转。")

# 指定要翻转图片的文件夹路径和输出文件夹路径
folder_path = 'image_test'

# 调用函数翻转图片
flip_images_in_folder(folder_path)