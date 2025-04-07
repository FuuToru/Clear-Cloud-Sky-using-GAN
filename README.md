# Clear Cloud Sky bằng GAN

**Clear Cloud Sky bằng GAN** là một dự án sử dụng Mạng Đối Kháng Tạo Sinh (Generative Adversarial Network - GAN) để loại bỏ mây khỏi ảnh vệ tinh, nâng cao độ rõ nét của chúng. Dự án được xây dựng dựa trên tập dữ liệu EuroSAT và cung cấp các công cụ để huấn luyện mô hình GAN cũng như thực hiện suy luận để tạo ra các ảnh không mây.

## Tính năng
- Huấn luyện mô hình GAN để loại bỏ mây khỏi ảnh vệ tinh.
- Thực hiện suy luận trên các ảnh mới bằng bộ tạo (generator) đã được huấn luyện.
- Thiết kế mô-đun với các siêu tham số có thể tùy chỉnh.

## Cài đặt

### Yêu cầu trước
- Python 3.8 trở lên
- PyTorch (khuyến nghị hỗ trợ GPU)
- Các thư viện phụ thuộc khác được liệt kê trong `requirements.txt`

### Thiết lập
Sao chép kho lưu trữ và cài đặt các gói cần thiết:
```bash
git clone https://github.com/your-username/fuutoru-gan-clear-cloud.git
cd fuutoru-gan-clear-cloud
pip install -r requirements.txt
```

## Hướng dẫn sử dụng

### 1. Chuẩn bị dữ liệu
Tải tập dữ liệu EuroSAT từ [Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset) và đặt nó vào thư mục `data/processed/`. Tham khảo `datasource/readme.txt` để biết thêm chi tiết.

### 2. Huấn luyện mô hình
Chạy tập lệnh huấn luyện với cấu hình được cung cấp:
```bash
python scripts/train.py --config config/train_config.yaml
```
- Sửa đổi `config/train_config.yaml` để điều chỉnh các siêu tham số như `batch_size`, `num_epochs` hoặc `dataset_path`.

### 3. Chạy suy luận
Tạo ảnh không mây bằng mô hình đã huấn luyện:
```bash
python scripts/inference.py --input_dir data/processed/ --output_dir outputs/images/ --checkpoint checkpoints/generator.pth
```
- `--input_dir`: Đường dẫn đến ảnh có mây.
- `--output_dir`: Đường dẫn để lưu ảnh được tạo ra.
- `--checkpoint`: Đường dẫn đến trọng số của bộ tạo đã huấn luyện.

## Kiến trúc GAN
Mô hình GAN trong dự án này bao gồm hai thành phần chính: **Bộ Tạo (Generator)** và **Bộ Phân Biệt (Discriminator)**, được thiết kế để phối hợp loại bỏ mây khỏi ảnh vệ tinh.

- **Bộ Tạo (Generator)**: Nhận một vector ẩn làm đầu vào và tạo ra một ảnh không mây giả lập. Nó sử dụng các lớp tích chập ngược (transposed convolution) để nâng cấp đầu vào thành ảnh RGB 64x64.
- **Bộ Phân Biệt (Discriminator)**: Đánh giá xem một ảnh là thật (không mây) hay giả (được tạo ra), sử dụng các lớp tích chập để giảm kích thước và phân loại đầu vào.
- **Hàm mất mát**: Sử dụng mất mát Entropy Chéo Nhị Phân (Binary Cross-Entropy - BCE) để huấn luyện cả hai mạng một cách đối kháng.

Kiến trúc được minh họa dưới đây:

![Kiến trúc GAN](/assets/GAN.png)

## Cấu trúc thư mục
```
fuutoru-gan-clear-cloud/
├── README.md                   # Tài liệu dự án
├── requirements.txt            # Các thư viện Python cần thiết
├── config/                     # Tệp cấu hình
│   └── train_config.yaml       # Siêu tham số huấn luyện
├── datasets/                   # Xử lý tập dữ liệu
│   └── sky_dataset.py          # Trình tải dữ liệu tùy chỉnh cho EuroSAT
├── datasource/                 # Thông tin nguồn dữ liệu
│   └── readme.txt              # Hướng dẫn tải tập dữ liệu EuroSAT
├── models/                     # Các thành phần GAN
│   ├── discriminator.py        # Mô hình phân biệt
│   ├── generator.py            # Mô hình tạo
│   └── losses.py               # Hàm mất mát cho huấn luyện GAN
├── scripts/                    # Tập lệnh thực thi
│   ├── inference.py            # Tập lệnh suy luận để loại bỏ mây
│   └── train.py                # Tập lệnh huấn luyện GAN
└── utils/                      # Các hàm tiện ích
    ├── logger.py               # Ghi log các số liệu huấn luyện
    ├── metrics.py              # Placeholder cho số liệu đánh giá
    └── visualizer.py           # Công cụ lưu và hiển thị ảnh
```

## Cấu hình
Tệp `train_config.yaml` chứa các tham số huấn luyện chính:
- `batch_size`: 64
- `image_size`: 64
- `num_epochs`: 100
- `learning_rate`: 0.0002
- `latent_dim`: 100
- `device`: cuda (hoặc cpu nếu không có GPU)

Điều chỉnh các cài đặt này dựa trên phần cứng và yêu cầu của bạn.

## Kết quả
- Nhật ký huấn luyện được lưu trong `outputs/metrics/log.txt`.
- Ảnh được tạo ra được lưu trong `outputs/images/` trong quá trình huấn luyện và suy luận.

## Đóng góp
Chúng tôi hoan nghênh mọi đóng góp! Để tham gia:
1. Fork kho lưu trữ.
2. Tạo một nhánh tính năng mới (`git checkout -b feature/your-feature`).
3. Cam kết thay đổi của bạn (`git commit -m "Thêm tính năng của bạn"`).
4. Đẩy lên nhánh của bạn (`git push origin feature/your-feature`).
5. Mở một Pull Request.

## Giấy phép
Dự án này được cấp phép theo Giấy phép MIT. Xem chi tiết trong tệp [LICENSE](LICENSE).

## Liên hệ
Để được hỗ trợ hoặc có thắc mắc, vui lòng mở một issue hoặc liên hệ qua [huutri231103@gmail.com](mailto:huutri231103@gmail.com).
