# Các thư viện cần cài đặt

Các gói cần cài đặt đã được ghi chú trong file 'requirement.txt'
Để cài đặt các gói này, sử dụng lệnh

```bash
pip install -r 'requirements.txt'
```

## Các tệp đính kèm gồm

[Data.csv:](Data.csv) Tệp lưu thông tin về số user và số movie

[rate.csv:](rate.csv) Tệp lưu thông tin về các rating

[Similarity.csv:](Similarity.csv) Tệp lưu ma trận tương đồng của các user từ các dữ liệu trước đó

[process.py:](process.py) Gồm các hàm để xử lý dữ liệu đầu vào

- process(): Đọc các dữ liệu đầu vào và khởi tạo đối tượng để gợi ý

- save_data(): Lưu lại các dữ liệu về rating sau mỗi lần sử dụng

[Memory_Base.py:](Memory_Base.py): class để đưa ra dự đoán và các chức năng

- Hàm khởi tạo và các hàm liên quan đến quá trình gợi ý: Người dùng không cần quan tâm đến các hàm này, đối tượng đã được khởi tạo thông qua hàm process() ở trên

- Hàm thay đổi dữ liệu: add_new_rating(user, movie, rating) và change_rating(user, movie, rating)
