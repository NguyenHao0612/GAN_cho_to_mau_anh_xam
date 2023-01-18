<h1 align="center">
  <img src="https://i.pinimg.com/originals/2d/fe/f3/2dfef32ec553b8fc45f8b5a5596c0ef0.jpg">
</h1>

<h3 align="center">SVTH: Nguyễn Trần Long Hảo 18DT2 106180079</h3>
<h3 align="center">SVTH: Nguyễn Văn Thương    18DT1 106180053</h3>
<h3 align="center">GVHD: TS. Hồ Phước Tiến</h3>

 `Kết quả đã thực hiện được`
  
 `Ground truth`
 -------------------->
 `Colorized output`

<p>
  <img src="https://github.com/NguyenHao0612/GAN_cho_to_mau_anh_xam/blob/main/static/ground_truth/grounf_truth.jpg" width="360">
  <img src="https://github.com/NguyenHao0612/GAN_cho_to_mau_anh_xam/blob/main/static/color/colorized_ouput_.jpg" width="360">
</p>
<p>
  <img src="https://github.com/NguyenHao0612/GAN_cho_to_mau_anh_xam/blob/main/static/ground_truth/grounf_truth_2.jpg" width="360">
  <img src="https://github.com/NguyenHao0612/GAN_cho_to_mau_anh_xam/blob/main/static/color/colorized_ouput_2.jpg" width="360">
</p>

## ✨ 1. Giới thiệu đề tài
>Đây là đồ án tốt nghiệp của tụi mình. Tụi mình sẽ xây dựng `2` mô hình. Mình phụ trách `mô hình 2` nên sẽ đề cập `mô hình 2` tại đây

>Hảo.

>[Link tải model](https://mega.nz/folder/9uhDyIYS#-YFNLBI9gts7H1AaittsYw)
```
"GAN cho tô màu ảnh xám"
```
Trong đề tài này mình sẽ xây dựng nên mô hình sử dụng cấu trúc GAN (Generative adversarial network) [Mạng đối nghịch tạo sinh].
Để tô màu được ảnh một ảnh xám (`1 kênh cường độ xám ở đây là kênh L`) sẽ được đưa vào, mô hình sẽ dự đoán giá trị màu của ảnh (`2 kênh màu ab`).
Sau đó sẽ ghép lại để cho ra bức ảnh hoàn chỉnh đầy màu sắc.

Mô hình Generator sẽ sử dụng kiến trúc Unet `Để tạo ra ảnh màu`

Mô hình Discriminator sẽ sử dụng các khối Conv-BatchNorm_LeakyReLU `Để phân biệt ảnh thật và ảnh giả`

>Nếu bạn đọc được bài này, và lần đầu bạn tìm hiểu.
Tại vì khi đó mình cũng tìm hiểu từ đầu, cũng có khá nhiều bài báo làm về đề tài này.
Bạn nên tìm hiểu thêm về (hay lắm >.<): 
`GAN conditional (GAN có điều kiện- trong bài này điều kiện là thêm ảnh xám vào)`
`Kiến trúc Unet`
`ResNet18`
`Kiến trúc Unet sử dụng ResNet18 làm phần mã hoá - Unet use ResNet18 backbone`
`PatchGAN Discriminator`
`fastai`

## 🚀 2. Generator
Generator được xây dựng dựa trên kiến trúc Unet

Huấn luyện trước Generator một cách có giám sát:
> ResNet18 được huấn luyện trước với nhiệm vụ phân loại trên tập dữ liệu ImageNet, được dùng cho phần mã hoá.
>
> Generator sẽ được huấn luyện với nhiệm vụ tô màu chỉ với mất mát L1.
> 
<img src="https://i.pinimg.com/originals/70/28/3b/70283b63ee9dbddd6a7b3e1ae3a3810e.png">
<p align="center">
  Cấu trúc ResNet18 cho phần mã hoá
<p>
<img src="https://i.pinimg.com/originals/02/37/88/0237880cbbaac091cfbfade19bbf5ae9.png">
<p align="center">
  Cấu trúc Phần mã hoá của Unet
<p>

Sau khi huấn luyện xong Generator. Thực hiện huấn luyện mô hình GAN như thông thường.

## 🚀 3. Discriminator
Kiến trúc của bộ phân biệt ( discriminator ) này thực hiện một mô hình bằng cách xếp chồng các khối `Conv-BatchNorm-LeackyReLU` để quyết định xem hình ảnh đầu vào là giả hay thật.
> Lưu ý rằng khối đầu tiên và khối cuối cùng không sử dụng chuẩn hóa và khối cuối cùng không có chức năng kích hoạt (nó được nhúng trong hàm mất mát sẽ sử dụng).

<img src="https://i.pinimg.com/originals/22/57/65/225765aa40af02268500b91e4cb9862b.png">
<p align="center">
  Cấu trúc Discriminator sử dụng PatchGAN Discriminator
<p>

> Ở đây sử dụng "Patch" Discriminator. Trong một vanilla discriminator, mô hình xuất ra một số (một tỷ lệ) đại diện cho mức độ mà mô hình nghĩ rằng đầu vào (là toàn bộ hình ảnh) là thật (hoặc giả). Trong patch discriminator, mô hình xuất ra một số cho mỗi patch 70 x 70 pixel của hình ảnh đầu vào và đối với mỗi patch sẽ quyết định xem nó có phải là giả hay không một cách riêng biệt. Sử dụng một mô hình như vậy cho nhiệm vụ chỉnh màu có vẻ hợp lý đối. Bởi vì những thay đổi cục bộ mà mô hình cần thực hiện thực sự quan trọng và có thể quyết định đến toàn bộ hình ảnh, như trong vanilla discriminator không thể quan tâm đến sự tinh tế của nhiệm vụ này. Ở đây, hình dạng đầu ra của mô hình là 30 x 30.

## 🚀 4. Hàm mất mát
  
`Tóm tắt thông số `

| Operator         | Description                                                         |
|------------------|---------------------------------------------------------------------|
| ``x``            | Ảnh xám (1 kênh thang độ xám ``L``)                                 |
| ``y``            | Giá trị màu gốc( 2 kênh màu ``a`` and ``b``)                        |
| ``G(x)``         | Giá trị màu được tạo ra từ Generator ( 2 kênh màu ``a`` and ``b``)  |
| ``D(x, y)``      | Đánh giá ảnh gốc là ảnh thật                                        |
| ``D(x, G(x))``   | Đánh giá ảnh được tạo ra từ Generator là thật hay giả               |
| ``D``            | Mô hình Discriminator                                               |
| ``G``            | Mô hình Generator                                                   |

<span style="display:none"></span>
  
<img src="https://i.pinimg.com/originals/72/33/35/723335e95dd8dd03050a4d0c6613a443.png">
  
<img src="https://i.pinimg.com/originals/0c/98/e0/0c98e03fc690e2afac4eae469e5391cd.png">

## ❤️ 5. Kết quả
  
<img src="https://i.pinimg.com/originals/20/78/3d/20783d9ab32b5f2a84e5fbf4a63f0e4d.png">
<p align="center">
  Đồ thị hàm mất mát
<p>
  
<p align="center">
  Kết quả tô màu
<p>
<img src="https://i.pinimg.com/originals/85/2c/fd/852cfd1adc94eb665b448ef6a32b2e60.jpg">
