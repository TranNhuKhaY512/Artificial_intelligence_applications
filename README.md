# Artificial_intelligence_applications
## Hệ Gợi Ý Căn Hộ, Biệt Thự Tại TP.Hồ Chí Minh
#### Đồ án môn học: Trí tuệ nhân tạo ứng dụng 

---

#### Giới thiệu dự án
- Dự án là một hệ thống thông minh có khả năng phân tích và đề xuất các bất động sản (căn hộ, biệt thự) phù hợp nhất cho người dùng tại khu vực Quận 1, Quận 2, Quận 7 và Quận Bình Thạnh thuộc TP.HCM.
- Hệ thống sử dụng giải thuật K-Nearest Neighbors (KNN) để phân tích độ tương đồng dựa trên các tiêu chí như mức giá, vị trí, diện tích, và số phòng ngủ. Bên cạnh đó, dự án được thiết kế dưới dạng ứng dụng Web Chatbot tích hợp Gemini API, giúp người dùng có thể tương tác và tìm kiếm bất động sản bằng ngôn ngữ tự nhiên.

---

#### Tính năng nổi bật
1. Tìm kiếm & Gợi ý thông minh (KNN): Áp dụng thuật toán KNN với độ đo khoảng cách Euclidean để tìm ra K=5 căn hộ có đặc điểm tương đồng nhất với truy vấn của người dùng.
2. Chatbot tương tác tự nhiên: Cho phép người dùng nhập yêu cầu bằng câu tự nhiên (VD: "Tôi cần tìm mua căn hộ 1 phòng ngủ tại quận 2 giá tầm 15 tỷ"). Hệ thống sử dụng NLP và Regex để bóc tách thông tin về giá, số phòng, và quận/huyện.
3. Phân tích truy vấn nâng cao: Kết hợp Gemini API để giải thích, phân tích chuyên sâu và đưa ra tư vấn thông minh dựa trên danh sách kết quả (VD: tư vấn căn nào đáng mua hơn dựa trên tiện ích, tầm view).
4. Tiền xử lý dữ liệu chuẩn xác: Sử dụng StandardScaler để chuẩn hóa các giá trị số (giá, số phòng ngủ) và OneHotEncoder để mã hóa khu vực địa lý thành vector nhị phân, giúp mô hình tính toán độ tương đồng một cách chính xác nhất.

---

#### Công nghệ sử dụng
- **Ngôn ngữ lập trình**: Python 3.11
- **Backend Framework**: Flask
- **Machine Learning**: scikit-learn (cài đặt NearestNeighbors, StandardScaler, OneHotEncoder)
- **Xử lý dữ liệu**: pandas, numpy
- **Xử lý chuỗi**: re (Biểu thức chính quy - Regex để tách dữ liệu)
- **Frontend**: HTML, CSS, JavaScript
- **AI API**: Google Gemini API

#### Cấu trúc dữ liệu (Dataset)
- Dữ liệu được thu thập thủ công và lưu trữ trong file ndat.csv chứa thông tin khoảng 200 bản ghi về các căn hộ và biệt thự.
- Các trường thông tin chính: Id, Tên, Địa chỉ, Giá, Số phòng ngủ, URL.
- Hình ảnh minh họa cho các bất động sản được lưu tại thư mục hinhanh/
- Dataset mẫu :
<img width="1238" height="272" alt="image" src="https://github.com/user-attachments/assets/418fc492-6eee-4013-bcd8-fe21f0679b67" />


---

#### Hướng dẫn cài đặt và chạy dự án
1. Clone repository:
```Bash
git clone [Nhập link GitHub dự án của bạn vào đây]
cd [Tên thư mục dự án]
```
2. Cài đặt các thư viện cần thiết:
Yêu cầu hệ thống cài đặt sẵn Python 3.11. Chạy lệnh sau để cài đặt các thư viện phụ thuộc:
```
Bash
pip install pandas numpy scikit-learn flask
```
3. Cấu hình API Key :
Tích hợp API Key của Google Gemini vào mã nguồn (phần Route /chat) để kích hoạt chức năng phân tích chuyên sâu của Chatbot.
4. Khởi động Server:
Chạy ứng dụng Flask bằng lệnh:
```
Bash
python app.py
```
Truy cập ứng dụng: Mở trình duyệt web và truy cập vào địa chỉ mặc định của Flask (thường là http://127.0.0.1:5000/) để trải nghiệm giao diện Chatbot.

---

## 📞 Thông tin liên hệ

Nếu bạn có bất kỳ câu hỏi nào về dự án hoặc muốn trao đổi thêm, vui lòng liên hệ với đại diện nhóm:

**Trần Như Khả Ý**
**Email:** trannhukhayy0512@gmail.com `\n`
**Điện thoại:** 0364551205 `\n`
**GitHub:** [TranNhuKhaY512](https://github.com/TranNhuKhaY512) `\n `
**LinkedIn:** [linkedin.com/in/trannhukhay051205](https://linkedin.com/in/trannhukhay051205) 
