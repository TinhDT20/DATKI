import torch
import os

# Đường dẫn đến tệp mô hình của bạn
model_path = "/home/tinhdt/apg_trajectory_tracking/trained_models/cartpole/test/model_cartpole1"

# Kiểm tra xem tệp có tồn tại không
if os.path.exists(model_path):
    try:
        # Tải mô hình từ tệp
        model = torch.load(model_path, map_location=torch.device('cpu'))  # Đưa mô hình về CPU để tránh lỗi nếu không có GPU
        print("Mô hình đã tải thành công!")
        
        # In thông tin chi tiết của mô hình
        print("\nThông tin mô hình:")
        print(model)
        
        # Nếu mô hình là dạng dictionary, in các key
        if isinstance(model, dict):
            print("\nCác thành phần trong mô hình:")
            for key in model.keys():
                print(f"- {key}")
        else:
            print("\nMô hình không phải là dạng dictionary. Đây là dạng:")
            print(type(model))
            
    except Exception as e:
        print(f"Có lỗi xảy ra khi tải mô hình: {e}")
else:
    print(f"Không tìm thấy tệp mô hình tại đường dẫn: {model_path}")

