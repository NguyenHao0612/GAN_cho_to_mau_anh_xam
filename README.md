<h1 align="center">GAN cho tô màu ảnh xám</h1>
<h3 align="center">SVTH: Nguyễn Trần Long Hảo 18DT2 106180079</h3>
<h3 align="center">SVTH: Nguyễn Văn Thương    18DT1 106180053</h3>
<h3 align="center">GVHD: TS. Hồ Phước Tiến</h3>
<p align="center">

  `Kết quả đã thực hiện được`
  
  `Ground truth`
  -------------------->
  `Colorized output`
</p>
<p>
  <img src="https://github.com/NguyenHao0612/GAN_cho_to_mau_anh_xam/blob/main/static/ground_truth/grounf_truth.jpg" width="240">
  <img src="https://github.com/NguyenHao0612/GAN_cho_to_mau_anh_xam/blob/main/static/color/colorized_ouput_.jpg" width="240">
</p>
<p>
  <img src="https://github.com/NguyenHao0612/GAN_cho_to_mau_anh_xam/blob/main/static/ground_truth/grounf_truth_2.jpg" width="240">
  <img src="https://github.com/NguyenHao0612/GAN_cho_to_mau_anh_xam/blob/main/static/color/colorized_ouput_2.jpg" width="240">
</p>

## ✨ 1. Giới thiệu đề tài
>Đây là đồ án tốt nghiệp của tụi mình. Tụi mình sẽ xây dựng `2` mô hình. Mình phụ trách `mô hình 2` nên sẽ đề cập `mô hình 2` tại đây

>Hảo.

>Link chứa model: `https://mega.nz/folder/9uhDyIYS#-YFNLBI9gts7H1AaittsYw`
```
"GAN cho tô màu ảnh xám"
```
Trong đề tài này mình sẽ xây dựng nên mô hình sử dụng cấu trúc GAN (Generative adversarial network) [Mạng đối nghịch tạo sinh].
Để tô màu được ảnh một ảnh xám (`1 kênh cường độ xám ở đây là kênh L`) sẽ được đưa vào, mô hình sẽ dự đoán giá trị màu của ảnh (`2 kênh màu ab`).
Sau đó sẽ ghép lại để cho ra bức ảnh hoàn chỉnh đầy màu sắc.

Mô hình Generator sẽ sử dụng kiến trúc Unet `Để tạo ra ảnh màu`

Mô hình Discriminator sẽ sử dụng các khối Conv-BatchNorm_LeakyReLU `Để phân biệt ảnh thật và ảnh giả`

>Nếu bạn đọc được bài này, và lần đầu bạn tìm hiểu về đề tài.
Tại vì khi đó mình cũng tìm hiểu từ đầu, cũng có khá nhiều bài báo làm về đề tài này.
Mình khuyên bạn nên tìm hiểu thêm về (hay lắm >.<): 
`GAN conditional (GAN có điều kiện- trong bài này điều kiện là thêm ảnh xám vào)`
`Kiến trúc Unet`
`ResNet18`
`Kiến trúc Unet sử dụng ResNet18 làm phần mã hoá - Unet use ResNet18 backbone`
`PatchGAN Discriminator`
`fastai`

## 🚀 2. Generator

## 🚀 3. Discriminator

## 🚀 4. Hàm mất mát

## ❤️ 5. Kết quả
