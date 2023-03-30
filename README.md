## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---

Các mục tiêu / bước của dự án này như sau:

* Tính toán ma trận hiệu chỉnh máy ảnh và các hệ số biến dạng được cung cấp một tập hợp các hình ảnh bàn cờ.
* Áp dụng hiệu chỉnh biến dạng cho ảnh thô.
* Áp dụng biến đổi phối cảnh để điều chỉnh hình ảnh nhị phân ("chế độ xem mắt chim").
* Sử dụng các phép biến đổi màu, độ dốc, v.v. để tạo hình ảnh nhị phân theo ngưỡng.
* Phát hiện pixel làn đường và điều chỉnh để tìm ranh giới làn đường.
* Xác định độ cong của làn đường và vị trí xe đối với tâm.
* Làm cong ranh giới làn đường được phát hiện trở lại hình ảnh ban đầu.
* Đầu ra hiển thị trực quan ranh giới làn đường và ước tính bằng số về độ cong của làn đường và vị trí phương tiện.

Hình ảnh để hiệu chỉnh máy ảnh được lưu trữ trong thư mục có tên `camera_cal`. Các hình ảnh trong `test_images` là để kiểm tra quy trình của bạn trên các khung hình đơn lẻ.

Video `challenge_video.mp4` là một video test bổ sung (và không bắt buộc) dành cho bạn nếu bạn muốn kiểm tra quy trình của mình trong các điều kiện phức tạp hơn một chút. Video `harder_challenge.mp4` là một video test khó hơn nhiều lần!

https://www.youtube.com/watch?v=iRTuCYx6quQ
