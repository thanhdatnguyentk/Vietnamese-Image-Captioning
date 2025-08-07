import json
import os
import requests
from tqdm import tqdm
import concurrent.futures # Để tải ảnh song song, tăng tốc độ

# --- Cấu hình đường dẫn và tên file JSON ---
# Thay đổi các đường dẫn này cho phù hợp với cấu trúc thư mục của bạn
DATA_DIR = './UIT-ViIC'
ANNOTATIONS_FILE = os.path.join(DATA_DIR, 'annotations', 'uitviic_captions_val2017.json') # Thay đổi nếu là val/test

# Thư mục đích để lưu ảnh. Đảm bảo nó khớp với cấu trúc bạn muốn.
# Ví dụ: nếu ANNOTATIONS_FILE là captions_train2017.json, thì thư mục đích nên là train2017
OUTPUT_IMAGE_DIR = os.path.join(DATA_DIR, 'images', 'val2017')

# --- Hàm tải một file ảnh ---
def download_image(image_info, output_dir):
    file_name = image_info['file_name']
    image_url = image_info['coco_url']
    output_path = os.path.join(output_dir, file_name)

    # Kiểm tra xem file đã tồn tại chưa để tránh tải lại
    if os.path.exists(output_path):
        return f"Skipped: {file_name} (already exists)"

    try:
        response = requests.get(image_url, stream=True, timeout=10) # Thêm timeout
        response.raise_for_status() # Kiểm tra lỗi HTTP

        # Ghi nội dung ảnh vào file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return f"Downloaded: {file_name}"
    except requests.exceptions.RequestException as e:
        return f"Failed to download {file_name} from {image_url}: {e}"
    except Exception as e:
        return f"An unexpected error occurred with {file_name}: {e}"

# --- Hàm chính để xử lý quá trình tải xuống ---
def download_uit_viic_images(annotations_file, output_image_dir):
    print(f"Loading annotations from: {annotations_file}")
    try:
        with open(annotations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        images_to_download = data['images']
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {annotations_file}")
        return
    except KeyError:
        print("Error: 'images' key not found in the annotation file. Please check the JSON structure.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {annotations_file}. Check file format.")
        return

    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(output_image_dir, exist_ok=True)
    print(f"Images will be saved to: {output_image_dir}")
    print(f"Found {len(images_to_download)} images to process.")

    # Sử dụng ThreadPoolExecutor để tải ảnh song song
    # Bạn có thể điều chỉnh max_workers tùy thuộc vào CPU và băng thông mạng
    MAX_WORKERS = 20 # Số lượng luồng tải xuống đồng thời
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Gửi các tác vụ tải xuống và hiển thị tiến độ bằng tqdm
        future_to_image = {executor.submit(download_image, img_info, output_image_dir): img_info for img_info in images_to_download}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_image), total=len(images_to_download), desc="Downloading images"):
            img_info = future_to_image[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                results.append(f"Image {img_info['file_name']} generated an exception: {exc}")
    
    # In ra báo cáo tóm tắt
    downloaded_count = sum(1 for r in results if "Downloaded" in r)
    skipped_count = sum(1 for r in results if "Skipped" in r)
    failed_count = sum(1 for r in results if "Failed" in r or "exception" in r)

    print("\n--- Download Summary ---")
    print(f"Total images processed: {len(images_to_download)}")
    print(f"Successfully downloaded: {downloaded_count}")
    print(f"Skipped (already exists): {skipped_count}")
    print(f"Failed to download: {failed_count}")

    if failed_count > 0:
        print("\n--- Failed Downloads Details ---")
        for r in results:
            if "Failed" in r or "exception" in r:
                print(r)

# --- Chạy script ---
if __name__ == "__main__":
    download_uit_viic_images(ANNOTATIONS_FILE, OUTPUT_IMAGE_DIR)

    # Nếu bạn có các file JSON khác (ví dụ: val, test), bạn có thể lặp lại:
    # download_uit_viic_images(os.path.join(DATA_DIR, 'annotations', 'captions_val2017.json'), os.path.join(DATA_DIR, 'images', 'val2017'))
    # download_uit_viic_images(os.path.join(DATA_DIR, 'annotations', 'captions_test2017.json'), os.path.join(DATA_DIR, 'images', 'test2017'))