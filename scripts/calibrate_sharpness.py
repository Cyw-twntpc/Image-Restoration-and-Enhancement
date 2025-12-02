# calibrate_sharpness.py
# 目的：分析指定資料夾中所有圖片的清晰度，同時使用兩種演算法（Laplacian, Sobel），
#       並生成一份包含兩種分數的 CSV 報告，以幫助使用者找到最佳的過濾方法與閾值。

import os
import cv2
import numpy as np
import csv
import subprocess
import platform
import argparse
import sys

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from photo_enhancer.data_preparation import get_image_sharpness as get_sharpness_sobel

def get_sharpness_laplacian(image: np.ndarray) -> float:
    """方法一：使用拉普拉斯變異數計算清晰度分數。"""
    if image is None:
        return 0.0
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def create_sharpness_report(source_dir, report_path):
    """遍歷資料夾，計算兩種清晰度分數，並寫入 CSV 報告。"""
    results = []
    print(f"正在分析資料夾: {os.path.abspath(source_dir)}")
    
    # Limit to first 2000 images for performance
    file_list = [f for f in os.listdir(source_dir) if f.lower().endswith('.png')][:2000]
    total = len(file_list)
    if total == 0:
        print("錯誤：在來源資料夾中找不到任何 .png 檔案。")
        return

    for i, filename in enumerate(file_list):
        file_path = os.path.join(source_dir, filename)
        try:
            image = cv2.imread(file_path)
            if image is not None:
                laplacian_score = get_sharpness_laplacian(image)
                sobel_score = get_sharpness_sobel(image)
                results.append([filename, laplacian_score, sobel_score])
                progress = ((i + 1) / total) * 100
                print(f"\r進度: {progress:.2f}% ({i+1}/{total}) - {filename}: Laplacian={laplacian_score:.2f}, Sobel={sobel_score:.2f}", end="")
            else:
                print(f"\r警告：讀取檔案失敗 {filename}")
        except Exception as e:
            print(f"\r處理檔案出錯 {filename}: {e}")
    
    print("\n分析完成，正在寫入報告...")

    # 寫入 CSV
    try:
        with open(report_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'sharpness_laplacian', 'sharpness_sobel'])
            results.sort(key=lambda x: x[2], reverse=True) # Sort by Sobel score
            writer.writerows(results)
        print(f"成功生成報告: {os.path.abspath(report_path)}")
        return True
    except Exception as e:
        print(f"寫入報告失敗: {e}")
        return False

def open_file(filepath):
    """跨平台開啟檔案。"""
    try:
        if platform.system() == 'Darwin':       # macOS
            subprocess.call(['open', filepath])
        elif platform.system() == 'Windows':    # Windows
            os.startfile(filepath)
        else:                                   # linux variants
            subprocess.call(['xdg-open', filepath])
    except Exception as e:
        print(f"自動開啟檔案失敗，請手動開啟。錯誤: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze image sharpness and generate a CSV report.")
    parser.add_argument("--source_dir", type=str, default="./data/hr/blur", help="Directory containing the images to analyze.")
    parser.add_argument("--report_filename", type=str, default="sharpness_report.csv", help="Output CSV report filename.")
    args = parser.parse_args()

    if create_sharpness_report(args.source_dir, args.report_filename):
        open_file(args.report_filename)
