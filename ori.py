from diffusers import StableDiffusionPipeline
import torch
import pandas as pd
import os

# Excel文件路径和模型路径
excel_file = "/disk1/fjm/LLM/xlsx/0707_secend.xlsx"
model_id = "/disk1/fujm/stable-diffusion-v1-5"

# 输出图片文件夹
output_dir = "./output_images"
os.makedirs(output_dir, exist_ok=True)

# 读取Excel（假如'prompt'是含有提示词的列名，按需修改）
df = pd.read_excel(excel_file)

# 指定开始和结束行（假设生成第6至第10行, 包含第6行和第10行）
start_row = 1  # 这里是第6行（因为iloc从0开始，这里是row[5]）
end_row   = 20  # 这里是第10行

# 初始化模型
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cuda")

for idx in range(start_row, end_row + 1):  # 包头包尾
    prompt = df.iloc[idx]['sentence']  # 改成你实际的列名
    image = pipe(prompt).images[0]
    out_name = f"row_{idx}.png"
    out_path = os.path.join(output_dir, out_name)
    image.save(out_path)
    print(f"已保存图片: {out_path}")