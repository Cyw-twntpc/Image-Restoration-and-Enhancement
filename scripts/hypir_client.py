from gradio_client import Client
import sys
import os
import shutil

client = Client("http://127.0.0.1:2281/")
lr_dir = "R:\\"
hypir_dir = r"D:\AI application\HYPIR"
save_dir = r".\HYPIR_pic_1"

file_list = os.listdir(lr_dir)
total = len(file_list)
count = 0

def move_and_rename_image(source_folder, destination_folder, new_filename):
    try:
        if not os.path.isdir(source_folder):
            print(f"錯誤：來源資料夾 '{source_folder}' 不存在。")
            return False

        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            print(f"已建立目標資料夾：'{destination_folder}'")

        files = os.listdir(source_folder)
        
        if not files:
            print("錯誤：來源資料夾中沒有任何檔案。")
            return False
        
        old_filename = files[0]
        source_path = os.path.join(source_folder, old_filename)
        
        destination_path = os.path.join(destination_folder, new_filename)

        shutil.move(source_path, destination_path)
        return True

    except IndexError:
        print("錯誤：來源資料夾中沒有任何檔案。")
        return False
    except FileNotFoundError:
        print(f"錯誤：找不到檔案或路徑。來源：'{source_path}', 目標：'{destination_path}'")
        return False
    except Exception as e:
        print(f"移動檔案時發生未知錯誤：{e}")
        return False


for filename in file_list:
    file_path = os.path.join(os.path.abspath(lr_dir), filename)
    input_data = {
        "path": file_path,
        "url": None,
        "size": os.path.getsize(file_path),
        "orig_name": filename,
        "mime_type": 'image/png',
        "is_stream": False,
        "meta": {}
    }
    result = client.predict(
            image=input_data,
            prompt="highly detailed, realistic skin texture, natural skin tone, no blur, no noise, no compression artifacts, professional portrait, 8K UHD, sharp features, ultra-realistic",
            upscale=2,
            patch_size=512,
            stride=256,
            seed=-1,
            api_name="/process"
    )
    move_and_rename_image(os.path.join(hypir_dir, "output"), os.path.abspath(save_dir), filename)
    count += 1
    progress = count / total * 100
    sys.stdout.write(f"\r進度: {progress:.2f}% (已處理 {count}/{total})")
    sys.stdout.flush()

