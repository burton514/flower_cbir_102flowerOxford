# Flower CBIR Workbench

Bản tóm tắt ngắn gọn những gì đã chốt cho project CBIR ảnh hoa theo hướng **feature-based**.

## 1) Mục tiêu
Xây dựng hệ thống tìm kiếm ảnh hoa tương tự với:
- tiền xử lí hợp lí, đúng thực tế
- bật/tắt từng feature
- chọn `Cosine`, `L2` hoặc `Chi-square` đúng điều kiện cho từng feature
- histogram được L1-normalize; non-histogram dùng z-score
- đánh giá truy hồi và độ tách lớp theo nhãn
- có giao diện để quan sát kết quả trung gian

---

## 2) Tư tưởng chính đã chốt
- **Feature không tốt tuyệt đối**; hiệu quả của feature phụ thuộc mạnh vào **pipeline tiền xử lí**.
- Vì vậy, hệ thống phải tách rõ:
  - **preprocessing chung**
  - **xử lí riêng nằm trong từng hàm trích xuất feature**
- Dataset hoa của bài này nên dùng hướng **object-centric preprocessing**, không dùng center-crop cố định theo giữa ảnh.

---

## 3) Pipeline tiền xử lí chuẩn
Luồng đã chốt:
1. Đọc ảnh gốc
2. Xóa nền
3. Tạo `mask` từ foreground
4. Làm sạch `mask`
5. Crop theo **bounding box của bông hoa**
6. Đưa bông về gần tâm ảnh
7. Chuẩn hoá mức chiếm khung hình
8. Đặt lại lên canvas `256x256`
9. Giữ ảnh màu gần gốc, chỉ chuẩn hoá sáng nhẹ nếu thật sự cần

### Lý do chọn hướng này
Vì dataset có thể còn:
- lệch tâm nhẹ
- chênh lệch kích thước bông hoa nhẹ
- sai khác do crop/resize cũ

Pipeline trên giúp giảm ảnh hưởng của:
- vị trí
- tỉ lệ object
- phần nền còn sót

---

## 4) Quy tắc trích xuất feature
- Có **preprocessing chung** cho toàn hệ thống.
- Mỗi feature có **xử lí riêng trong chính hàm extract**.
- Sau khi đổi cấu hình feature, phải có nút:
  - **Áp dụng cấu hình**
  - **Trích xuất đặc trưng**
- Chỉ sau khi bấm trích xuất lại thì hệ thống mới dùng bộ đặc trưng mới.

---

## 5) Feature ứng viên trong hệ thống

### A. Nhóm màu
- HSV Histogram
- RGB Histogram
- Hue Histogram
- Dominant Colors
- Color Moments
- Lab Moments / Lab Histogram

### B. Nhóm hình dáng
- Hu Moments
- Geometric Shape Features
- Contour Basic Descriptors
- Radial Signature
- Fourier Shape Descriptor
- Symmetry Score

### C. Nhóm texture / gradient
- LBP
- GLCM
- HOG

### D. Nhóm biên / cấu trúc
- Edge Histogram / Edge Orientation Histogram
- Canny-derived Features
- Sobel / Gradient Histogram
- Petal-edge profile (mở rộng)

### E. Nhóm đặc trưng cục bộ
- SIFT
- ORB
- AKAZE
- BRISK

### F. Meta / kiểm tra pipeline
- Foreground Occupancy
- Centroid Offset
- Mask Quality Indicators

### Ghi chú về Canny
**Canny không phải feature cuối cùng độc lập.**
Canny được dùng để:
- hiển thị edge map trung gian
- tạo đầu vào cho feature biên / contour / cấu trúc

---

## 6) Quy tắc distance, chuẩn hoá, fusion
- Mỗi feature có thể chọn riêng:
  - `Cosine`
  - `L2`
- Chuẩn hoá dùng **z-score**.
- Khi bật nhiều feature cùng lúc:
  - tính distance **riêng cho từng feature**
  - sau đó gộp theo **weight**
- Có 2 chế độ weight:
  - **Auto weight** để user ít thao tác
  - **Manual weight** nếu muốn tự chỉnh

---

## 7) Đánh giá
### Retrieval metrics
- Precision@5
- MAP@5
- MRR@5

### Độ tách lớp theo nhãn
- Mean intra-class distance
- Mean inter-class distance
- Separation ratio
- Silhouette score
- Fisher discriminant ratio

Yêu cầu xuyên suốt: **chỉ dùng công thức chính thống, không tự bịa**.

---

## 8) Kiến trúc hệ thống
### Offline pipeline
Dùng để:
- tiền xử lí toàn bộ dataset
- trích xuất feature hàng loạt
- tính thống kê z-score
- lưu vector vào SQLite

### Online pipeline
Dùng để:
- truy vấn ảnh trong dataset
- truy vấn ảnh ngoài dataset
- áp dụng đúng preprocessing như offline
- trích feature theo cấu hình hiện hành
- trả về top-5 ảnh tương tự

---

## 9) Giao diện mong muốn
App cần có:
- bật/tắt feature
- chọn `Cosine / L2` cho từng feature
- chỉnh weight
- xem ảnh trung gian ở từng bước
- nút **Áp dụng cấu hình**
- nút **Trích xuất đặc trưng**
- nút **Đánh giá**
- khu vực **Truy vấn**

---

## 10) Lưu trữ
- Toàn bộ dữ liệu đặc trưng dùng **SQLite**
- Config hệ thống dùng **JSON**
- Nhãn lấy từ **prefix tên file** theo quy ước `nhãn_####`, ví dụ `1_0003.png` có nhãn `1`.

---

## 11) Kết luận ngắn
Hướng làm đúng cho bài này là:
- **chuẩn hoá theo bông hoa**, không chuẩn hoá theo giữa ảnh một cách máy móc
- dùng **preprocessing chung + xử lí riêng cho từng feature**
- xây hệ thống tổng hợp để so sánh feature công bằng, nhưng vẫn đủ linh hoạt để tối ưu tổ hợp feature cuối cùng


## Cập nhật tiền xử lí
- Đồng bộ online/offline qua cùng một `preprocess_image()`.
- Sửa tạo mask: chỉ dùng alpha thật; nếu ảnh chưa có alpha thì mới chạy `rembg`.
- Giữ component lớn nhất của mask để giảm nhiễu.
- Chuẩn hóa object theo mask thật và dán foreground lên canvas bằng mask, không paste cả nền.
- Lưu thêm ảnh debug trong `workspace/preprocessed/debug/`.

## Cập nhật rà soát logic lý thuyết
- `chi_square` chỉ dùng cho feature dạng histogram không âm đã L1-normalize. Non-histogram feature sẽ chỉ dùng `cosine` hoặc `l2`; nếu config cũ chọn sai, hệ thống tự fallback về `l2` khi lưu extraction run.
- Fusion đã đổi sang quy trình: tính distance riêng cho từng feature → min-max normalize distance trên tập ứng viên của feature đó → cộng theo weight. Cách này tránh feature có thang đo distance lớn áp đảo các feature khác.
- Evaluation retrieval dùng lại đúng cấu hình `fusion` đã được lưu trong extraction run, không còn cố định `auto_weight=True`.
- Query online dùng lại cấu hình preprocessing/fusion của extraction run mới nhất để tránh lệch pipeline nếu người dùng chỉnh config sau khi đã trích xuất.
- Hu Moments được tính trên mask nhị phân với `binaryImage=True` và log transform chuẩn hơn.
- GLCM được tính trong ROI foreground, chỉ đếm cặp pixel hợp lệ trong mask, tránh để nền trắng chi phối texture.
- Nhãn lớp hỗ trợ `auto`, `parent_folder`, `filename_prefix`; mặc định `auto` ưu tiên thư mục con của dataset, phù hợp dataset dạng `flowers/rose/*.jpg`.
- Truy vấn ảnh trong dataset đã sửa để loại chính ảnh query trước khi lấy đủ `top_k`, không còn bị thiếu kết quả.
- Danh sách ảnh dataset nhận cả extension viết hoa như `.JPG`, `.PNG`.
- Tên file output preprocessing có thêm đường dẫn tương đối và hash ngắn để tránh ghi đè khi nhiều lớp có ảnh trùng tên.
- Class separation cũng dùng lại logic distance/fusion đã normalize theo feature để tính intra/inter/silhouette; Fisher ratio vẫn được tính trên vector nối sau chuẩn hóa như một chỉ số phụ.


## Cập nhật bản hoàn thiện cho dataset `nhãn_####`
- Config mặc định đã đặt `label_source = filename_prefix`, phù hợp tên file dạng `1_0003.png`.
- Config mặc định đã tắt `rembg` vì dataset của bài đã được resize và xóa nền từ Oxford 102 Flowers. Nếu ảnh query ngoài dataset chưa xóa nền, có thể bật lại trong sidebar.
- Tab tiền xử lý có nút **Kiểm tra nhanh dataset** để thống kê số ảnh, số nhãn, kích thước ảnh, định dạng và số ảnh theo nhãn.
- Kết quả truy vấn hiển thị rõ `Distance` và `Similarity = 1 - Distance`; hệ thống xếp hạng theo distance tăng dần, tương đương similarity giảm dần.
- Query trả thêm bảng đóng góp từng đặc trưng: raw distance, normalized distance, weight và contribution. Bảng này dùng để trình bày kết quả trung gian của quá trình tìm kiếm theo yêu cầu đề bài.
