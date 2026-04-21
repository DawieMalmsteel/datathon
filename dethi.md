−→ ĐỀ THI VÒNG 1
DATATHON 2026
THE GRIDBREAKER

Breaking Business Boundaries

Hosted by
VinTelligence
VinUniversity Data Science & AI Club

Cuộc thi Khoa học Dữ liệu đầu tiên tại VinUniversity

Biến Dữ liệu thành Giải pháp cho Doanh nghiệp

DATATHON 2026 — The Gridbreakers

Được tổ chức bởi VinTelligence — VinUni DS&AI Club

Contents

1 Mô tả Dữ liệu

1.3.1
1.3.2
1.3.3
1.3.4

1.1 Giới thiệu . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
1.2 Tổng quan các bảng dữ liệu . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
1.3 Bảng Master
products.csv — Danh mục sản phẩm . . . . . . . . . . . . . . . . . . . .
customers.csv — Khách hàng . . . . . . . . . . . . . . . . . . . . . . . .
promotions.csv — Chương trình khuyến mãi
. . . . . . . . . . . . . . .
geography.csv — Địa lý . . . . . . . . . . . . . . . . . . . . . . . . . . .
1.4 Bảng Transaction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
orders.csv — Đơn hàng . . . . . . . . . . . . . . . . . . . . . . . . . . .
order_items.csv — Chi tiết đơn hàng . . . . . . . . . . . . . . . . . . .
payments.csv — Thanh toán . . . . . . . . . . . . . . . . . . . . . . . . .
shipments.csv — Vận chuyển . . . . . . . . . . . . . . . . . . . . . . . .
returns.csv — Trả hàng . . . . . . . . . . . . . . . . . . . . . . . . . . .
reviews.csv — Đánh giá . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
sales.csv — Dữ liệu doanh thu . . . . . . . . . . . . . . . . . . . . . . .
1.6 Bảng Operational . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
inventory.csv — Tồn kho . . . . . . . . . . . . . . . . . . . . . . . . . .
web_traffic.csv — Lưu lượng truy cập . . . . . . . . . . . . . . . . . .
1.7 Quan hệ giữa các bảng . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

1.4.1
1.4.2
1.4.3
1.4.4
1.4.5
1.4.6

1.5 Bảng Analytical

1.6.1
1.6.2

1.5.1

2 Đề Bài

2.1 Phần 1 — Câu hỏi Trắc nghiệm . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2.2 Phần 2 — Trực quan hoá và Phân tích Dữ liệu . . . . . . . . . . . . . . . . . . .
2.2.1 Yêu cầu . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2.3 Phần 3 — Mô hình Dự báo Doanh thu (Sales Forecasting) . . . . . . . . . . . . .
2.3.1 Bối cảnh kinh doanh . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2.3.2 Định nghĩa bài toán . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2.3.3 Dữ liệu . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2.3.4 Chỉ số đánh giá . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2.3.5 Định dạng file nộp . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2.3.6 Ràng buộc

3 Thang điểm Chấm thi

4 Hướng dẫn Nộp bài

4.1 Checklist nộp bài . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

1

2
2
2
2
2
3
3
3
3
3
4
4
4
4
4
5
5
5
5
5
6

7
7
10
10
11
11
11
11
11
11
12

13

16
16

DATATHON 2026 — The Gridbreakers

Được tổ chức bởi VinTelligence — VinUni DS&AI Club

Mô tả Dữ liệu

Giới thiệu

Bộ dữ liệu mô phỏng hoạt động của một doanh nghiệp thời trang thương mại điện tử tại Việt
Nam trong giai đoạn từ 04/07/2012 đến 31/12/2022. Dữ liệu bao gồm 15 file CSV, được
chia thành 4 lớp: Master (dữ liệu tham chiếu), Transaction (giao dịch), Analytical (phân tích)
và Operational (vận hành).

Phân chia dữ liệu cho bài toán dự báo:

• sales_train.csv: 04/07/2012 → 31/12/2022
• sales_test.csv: 01/01/2023 → 01/07/2024

Tổng quan các bảng dữ liệu

Table 1: Danh sách các file dữ liệu

# File

Lớp

Mô tả

1
2
3
4
5
6
7
8
9
10
11
12
13
14

Danh mục sản phẩm
Thông tin khách hàng
Các chiến dịch khuyến mãi
Danh sách mã bưu chính các vùng

Master
products.csv
Master
customers.csv
Master
promotions.csv
Master
geography.csv
Transaction Thông tin đơn hàng
orders.csv
Transaction Chi tiết từng dòng sản phẩm trong đơn
order_items.csv
Transaction Thông tin thanh toán tương ứng 1:1 với đơn hàng
payments.csv
Transaction Thông tin vận chuyển
shipments.csv
Transaction Các sản phẩm bị trả lại
returns.csv
Transaction Đánh giá sản phẩm sau giao hàng
reviews.csv
Analytical
sales.csv
sample_submission.csv Analytical
inventory.csv
web_traffic.csv

Dữ liệu doanh thu huấn luyện
Định dạng file nộp bài (mẫu)
Operational Ảnh chụp tồn kho cuối tháng
Operational Lưu lượng truy cập website hàng ngày

Bảng Master

▶ products.csv — Danh mục sản phẩm

Cột

Kiểu Mô tả

product_id
product_name
category
segment
size
color
price
cogs

int
str
str
str
str
str
float
float

Khoá chính
Tên sản phẩm
Danh mục sản phẩm
Phân khúc thị trường của sản phẩm
Kích cỡ sản phẩm
Nhãn màu sản phẩm
Giá bán lẻ
Giá vốn hàng bán

Ràng buộc: cogs < price với mọi sản phẩm.

2

DATATHON 2026 — The Gridbreakers

Được tổ chức bởi VinTelligence — VinUni DS&AI Club

▶ customers.csv — Khách hàng

Cột

Kiểu Mô tả

customer_id
zip
city
signup_date
gender
age_group
acquisition_channel

int
int
str
date
str
str
str

Khoá chính
Mã bưu chính (FK → geography.zip)
Tên thành phố của khách hàng
Ngày đăng ký tài khoản
Giới tính khách hàng (nullable)
Nhóm tuổi khách hàng (nullable)
Kênh tiếp thị khách hàng đăng ký qua (nullable)

▶ promotions.csv — Chương trình khuyến mãi

Cột

Kiểu Mô tả

promo_id
promo_name
promo_type
discount_value
start_date
end_date
applicable_category
promo_channel
stackable_flag
min_order_value

str
str
str
float
date
date
str
str
int
float

Khoá chính
Tên chiến dịch kèm năm
Loại giảm giá: theo phần trăm hoặc số tiền cố định
Giá trị giảm (phần trăm hoặc số tiền tùy promo_type)
Ngày bắt đầu chiến dịch
Ngày kết thúc chiến dịch
Danh mục áp dụng, null nếu áp dụng tất cả
Kênh phân phối áp dụng khuyến mãi (nullable)
Cờ cho phép áp dụng đồng thời nhiều khuyến mãi
Giá trị đơn hàng tối thiểu để áp dụng khuyến mãi (nullable)

Công thức giảm giá:

• percentage: discount_amount = quantity × unit_price × (discount_value/100)
• fixed: discount_amount = quantity × discount_value

▶ geography.csv — Địa lý

Cột

Kiểu Mô tả

zip
city
region
district

int
str
str
str

Khoá chính (mã bưu chính)
Tên thành phố
Vùng địa lý
Tên quận/huyện

Bảng Transaction

▶ orders.csv — Đơn hàng

Cột

Kiểu Mô tả

order_id
order_date
customer_id
zip
order_status
payment_method
device_type
order_source

int
date
int
int
str
str
str
str

Khoá chính
Ngày đặt hàng
FK → customers.customer_id
Mã bưu chính giao hàng (FK → geography.zip)
Trạng thái xử lý của đơn hàng
Phương thức thanh toán được sử dụng
Thiết bị khách hàng dùng khi đặt hàng
Kênh marketing dẫn đến đơn hàng

3

DATATHON 2026 — The Gridbreakers

Được tổ chức bởi VinTelligence — VinUni DS&AI Club

▶ order_items.csv — Chi tiết đơn hàng

Cột

Kiểu Mô tả

order_id
product_id
quantity
unit_price
discount_amount
promo_id
promo_id_2

int
int
int
float
float
str
str

FK → orders.order_id
FK → products.product_id
Số lượng sản phẩm đặt mua
Đơn giá sau khi áp dụng khuyến mãi
Tổng số tiền giảm giá cho dòng sản phẩm này
FK → promotions.promo_id (nullable)
FK → promotions.promo_id, khuyến mãi thứ hai (nullable)

▶ payments.csv — Thanh toán

Cột

Kiểu Mô tả

order_id
payment_method
payment_value
installments

int
str
float
int

FK → orders.order_id (quan hệ 1:1)
Phương thức thanh toán
Tổng giá trị thanh toán của đơn hàng
Số kỳ trả góp

▶ shipments.csv — Vận chuyển

Cột

Kiểu Mô tả

order_id
ship_date
delivery_date
shipping_fee

int
date
date
float

FK → orders.order_id
Ngày gửi hàng
Ngày giao hàng đến tay khách
Phí vận chuyển (0 nếu đơn được miễn phí)

Chỉ tồn tại cho đơn hàng có trạng thái shipped, delivered hoặc returned.

▶ returns.csv — Trả hàng

Cột

Kiểu Mô tả

return_id
order_id
product_id
return_date
return_reason
return_quantity
refund_amount

str
int
int
date
str
int
float

Khoá chính
FK → orders.order_id
FK → products.product_id
Ngày khách gửi trả hàng
Lý do trả hàng
Số lượng sản phẩm trả lại
Số tiền hoàn lại cho khách

▶ reviews.csv — Đánh giá

Cột

Kiểu Mô tả

review_id
order_id
product_id
customer_id
review_date
rating
review_title

str
int
int
int
date
int
str

Khoá chính
FK → orders.order_id
FK → products.product_id
FK → customers.customer_id
Ngày khách gửi đánh giá
Điểm đánh giá từ 1 đến 5
Tiêu đề đánh giá của khách hàng

4

DATATHON 2026 — The Gridbreakers

Được tổ chức bởi VinTelligence — VinUni DS&AI Club

Bảng Analytical

▶ sales.csv — Dữ liệu doanh thu

Cột

Kiểu Mô tả

Date
Revenue
COGS

date
float
float

Ngày đặt hàng
Tổng doanh thu thuần
Tổng giá vốn hàng bán

Split File

Khoảng thời gian

Train
Test

sales.csv
sales_test.csv

04/07/2012 – 31/12/2022
01/01/2023 – 01/07/2024

Lưu ý: Tập test sẽ không được công bố mà được dùng để đánh giá kết quả mô hình trên
Kaggle. Cấu trúc của file test sẽ giống với sample_submission.csv

Bảng Operational

▶ inventory.csv — Tồn kho

Cột

Kiểu Mô tả

snapshot_date
product_id
stock_on_hand
units_received
units_sold
stockout_days
days_of_supply
fill_rate
stockout_flag
overstock_flag
reorder_flag
sell_through_rate
product_name
category
segment
year
month

date
int
int
int
int
int
float
float
int
int
int
float
str
str
str
int
int

Ngày chụp ảnh tồn kho (cuối tháng)
FK → products.product_id
Số lượng tồn kho cuối tháng
Số lượng nhập kho trong tháng
Số lượng bán ra trong tháng
Số ngày hết hàng trong tháng
Số ngày tồn kho có thể đáp ứng nhu cầu bán
Tỷ lệ đơn hàng được đáp ứng đủ từ tồn kho
Cờ báo tháng có xảy ra hết hàng
Cờ báo tồn kho vượt mức cần thiết
Cờ báo cần tái đặt hàng sớm
Tỷ lệ hàng đã bán so với tổng hàng sẵn có
Tên sản phẩm
Danh mục sản phẩm
Phân khúc sản phẩm
Năm trích từ snapshot_date
Tháng trích từ snapshot_date

▶ web_traffic.csv — Lưu lượng truy cập

Cột

Kiểu Mô tả

date
sessions
unique_visitors
page_views
bounce_rate
avg_session_duration_sec
traffic_source

date
int
int
int
float
float
str

Ngày ghi nhận lưu lượng
Tổng số phiên truy cập trong ngày
Số lượt khách truy cập duy nhất
Tổng số lượt xem trang
Tỷ lệ phiên chỉ xem một trang rồi thoát
Thời gian trung bình mỗi phiên (giây)
Kênh nguồn dẫn traffic về website

5

DATATHON 2026 — The Gridbreakers

Được tổ chức bởi VinTelligence — VinUni DS&AI Club

Quan hệ giữa các bảng

Table 2: Quy tắc quan hệ (Cardinality)

Quan hệ

Cardinality

orders ↔ payments
orders ↔ shipments
orders ↔ returns
orders ↔ reviews
order_items ↔ promotions
products ↔ inventory

1 : 1
1 : 0 hoặc 1 (trạng thái shipped/delivered/returned)
1 : 0 hoặc nhiều (trạng thái returned)
1 : 0 hoặc nhiều (trạng thái delivered, ∼20%)
nhiều : 0 hoặc 1
1 : nhiều (1 dòng/sản phẩm/tháng)

6

DATATHON 2026 — The Gridbreakers

Được tổ chức bởi VinTelligence — VinUni DS&AI Club

Đề Bài

Phần 1 — Câu hỏi Trắc nghiệm

Chọn một đáp án đúng nhất cho mỗi câu hỏi. Các câu hỏi yêu cầu tính toán trực tiếp từ
dữ liệu được cung cấp.

Q1. Trong số các khách hàng có nhiều hơn một đơn hàng, trung vị số ngày giữa hai lần
mua liên tiếp (inter-order gap) xấp xỉ là bao nhiêu? (Tính từ orders.csv)

A) 30 ngày

B) 90 ngày

C) 144 ngày

D) 365 ngày

Q2. Phân khúc sản phẩm (segment) nào trong products.csv có tỷ suất lợi nhuận gộp
trung bình cao nhất, với công thức (price − cogs)/price?

A) Premium

B) Performance

C) Activewear

D) Standard

Q3. Trong các bản ghi trả hàng liên kết với sản phẩm thuộc danh mục Streetwear (join
returns với products theo product_id), lý do trả hàng nào xuất hiện nhiều nhất?

A) defective

B) wrong_size

C) changed_mind

D) not_as_described

Q4. Trong web_traffic.csv, nguồn truy cập (traffic_source) nào có tỷ lệ thoát trung
bình (bounce_rate) thấp nhất trên tất cả các ngày xuất hiện nguồn đó trong cột traffic_source?

A) organic_search

B) paid_search

C) email_campaign

D) social_media

7

DATATHON 2026 — The Gridbreakers

Được tổ chức bởi VinTelligence — VinUni DS&AI Club

Q5. Tỷ lệ phần trăm các dòng trong order_items.csv có áp dụng khuyến mãi (tức là promo_id
không null) xấp xỉ là bao nhiêu?

A) 12%

B) 25%

C) 39%

D) 54%

Q6. Trong customers.csv, xét các khách hàng có age_group khác null, nhóm tuổi nào có số
đơn hàng trung bình trên mỗi khách hàng cao nhất? (tổng số đơn / số khách hàng trong
nhóm)

A) 55+

B) 25–34

C) 35–44

D) 45–54

Q7. Vùng (region) nào trong geography.csv tạo ra tổng doanh thu cao nhất trong
sales_train.csv?

A) West

B) Central

C) East

D) Cả ba vùng có doanh thu xấp xỉ bằng nhau

Q8. Trong các đơn hàng có order_status = ’cancelled’ trong orders.csv, phương thức
thanh toán nào được sử dụng nhiều nhất?

A) credit_card

B) cod

C) paypal

D) bank_transfer

Q9. Trong bốn kích thước sản phẩm (S, M, L, XL), kích thước nào có tỷ lệ trả hàng cao
nhất, được định nghĩa là số bản ghi trong returns chia cho số dòng trong order_items (join
với products theo product_id)?

A) S

B) M

C) L

8

DATATHON 2026 — The Gridbreakers

Được tổ chức bởi VinTelligence — VinUni DS&AI Club

D) XL

Q10. Trong payments.csv, kế hoạch trả góp nào có giá trị thanh toán trung bình trên
mỗi đơn hàng cao nhất?

A) 1 kỳ (trả một lần)

B) 3 kỳ

C) 6 kỳ

D) 12 kỳ

9

DATATHON 2026 — The Gridbreakers

Được tổ chức bởi VinTelligence — VinUni DS&AI Club

Phần 2 — Trực quan hoá và Phân tích Dữ liệu

Khám phá bộ dữ liệu để tìm ra các insight có ý nghĩa kinh doanh. Phần này được đánh
giá dựa trên tính sáng tạo, chiều sâu phân tích và chất lượng trình bày. Không có
đáp án đúng duy nhất — ban giám khảo đánh giá khả năng kể chuyện bằng dữ liệu (data
storytelling) của các đội.

▶ Yêu cầu
Các đội thi tự do lựa chọn góc nhìn phân tích từ bộ dữ liệu. Bài nộp cần bao gồm hai thành
phần:

1. Trực quan hoá (Visualizations): Tạo các biểu đồ, đồ thị, bản đồ hoặc dashboard trực
quan để thể hiện các pattern, xu hướng và mối quan hệ trong dữ liệu. Mỗi hình ảnh cần
có tiêu đề, nhãn trục rõ ràng và chú thích phù hợp.

2. Phân tích (Analysis): Viết phần giải thích đi kèm mỗi trực quan hoá, bao gồm:

• Mô tả những gì biểu đồ thể hiện và tại sao góc nhìn này quan trọng
• Các phát hiện chính (key findings) được hỗ trợ bởi số liệu cụ thể
• Ý nghĩa kinh doanh (business implications) hoặc đề xuất hành động (actionable rec-

ommendations)

Tiêu chí đánh giá Phần 2: Bài nộp được đánh giá theo bốn cấp độ phân tích. Cấp độ
cao hơn bao gồm và nâng cao cấp độ thấp hơn.

Cấp độ

Câu hỏi

Ban giám khảo đánh giá

Descriptive What happened?

Diagnostic Why did it hap-
pen?
Predictive What is likely to

happen?

Prescriptive What

should we

do?

Thống kê tổng hợp chính xác, biểu đồ có
nhãn rõ ràng, tổng hợp dữ liệu đúng
Giả thuyết nhân quả, so sánh phân khúc,
xác định bất thường có bằng chứng hỗ trợ
Ngoại suy xu hướng, phân tích tính mùa
vụ, phân tích chỉ số dẫn xuất
Đề xuất hành động kinh doanh được hỗ
trợ bởi dữ liệu; đánh đổi được định lượng

Các đội đạt cấp độ Prescriptive nhất quán trên nhiều phân tích sẽ đạt điểm cao nhất.

10

DATATHON 2026 — The Gridbreakers

Được tổ chức bởi VinTelligence — VinUni DS&AI Club

Phần 3 — Mô hình Dự báo Doanh thu (Sales Forecasting)

▶ Bối cảnh kinh doanh
Bạn là nhà khoa học dữ liệu tại một công ty thương mại điện tử thời trang Việt Nam. Doanh
nghiệp cần dự báo nhu cầu chính xác ở mức chi tiết để tối ưu hoá phân bổ tồn kho, lập kế hoạch
khuyến mãi và quản lý logistics trên toàn quốc.

▶ Định nghĩa bài toán
Dự báo cột Revenue trong khoảng thời gian của sales_test.csv.

Mỗi dòng trong tập test là một bộ (Date,Revenue,COGS) duy nhất trong giai đoạn 01/01/2023

– 01/07/2024.

▶ Dữ liệu

Split File

Khoảng thời gian

Train sales.csv
Test

sales_test.csv

04/07/2012 – 31/12/2022
01/01/2023 – 01/07/2024

▶ Chỉ số đánh giá
Bài nộp được đánh giá bằng ba chỉ số:
Mean Absolute Error (MAE):

MAE =

1
n

n
X

i=1

|Fi − Ai|

Root Mean Squared Error (RMSE):

RMSE =

v
u
u
t

1
n

n
X

(Fi − Ai)2

i=1

R2 (Coefficient of Determination):

R2 = 1 −

Pn

Pn

i=1(Ai − Fi)2
i=1(Ai − ¯A)2

trong đó Fi là giá trị dự báo, Ai là giá trị thực, ¯A là trung bình giá trị thực. MAE đo độ
lệch tuyệt đối trung bình, RMSE phạt nặng hơn các sai số lớn, và R2 thể hiện tỷ lệ phương sai
được giải thích bởi mô hình.

MAE và RMSE càng thấp càng tốt. R2 càng cao càng tốt (lý tưởng gần 1).

▶ Định dạng file nộp
Nộp file submission.csv với các cột sau:

Các dòng trong submission.csv phải giữ đúng thứ tự như sample_submission.csv.
Không sắp xếp lại hoặc xáo trộn.

Ví dụ:

11

DATATHON 2026 — The Gridbreakers

Được tổ chức bởi VinTelligence — VinUni DS&AI Club

Date,Revenue,COGS
2023-01-01,26607.2,2585.15
2023-01-02,1007.89,163.0
2023-01-03,1089.51,821.12
...

▶ Ràng buộc

1. Không dùng dữ liệu ngoài: Tất cả đặc trưng phải được tạo từ các file dữ liệu được

cung cấp.

2. Tính tái lập (Reproducibility): Đính kèm toàn bộ mã nguồn. Đặt random seed khi

cần thiết.

3. Khả năng giải thích (Explainability): Trong report, bao gồm một mục giải thích các
yếu tố dẫn động doanh thu chính được mô hình xác định (vd: feature importances, SHAP
values, hoặc partial dependence plots). Giải thích những gì mô hình học được bằng ngôn
ngữ kinh doanh.

12

DATATHON 2026 — The Gridbreakers

Được tổ chức bởi VinTelligence — VinUni DS&AI Club

Thang điểm Chấm thi

Tổng điểm tối đa: 100 điểm, phân bổ theo ba phần thi. Điểm thành phần không làm
tròn cho đến khi tính tổng cuối cùng.

Table 3: Phân bổ điểm tổng quan

Phần Nội dung

Điểm Tỷ trọng

1
2
3

Câu hỏi Trắc nghiệm (MCQ)
Trực quan hoá & Phân tích (EDA)
Mô hình Dự báo Doanh thu (Forecasting)

20
60
20

20%
60%
20%

Tổng

100

100%

Phần 1 — Câu hỏi Trắc nghiệm (20 điểm)

Mỗi câu đúng được 2 điểm. Không trừ điểm cho câu trả lời sai.

Thành phần

Số câu

Điểm

Câu trả lời đúng
Câu trả lời sai
Không trả lời

10 câu
—
—

2 điểm / câu
0 điểm
0 điểm

Tổng tối đa

20 điểm

Phần 2 — Trực quan hoá & Phân tích EDA (60 điểm)

Phần này được chấm theo bốn tiêu chí độc lập, mỗi tiêu chí tương ứng với một cấp độ phân
tích trong rubric. Ban giám khảo chấm từng tiêu chí trên thang điểm thành phần, sau đó cộng
lại.

13

DATATHON 2026 — The Gridbreakers

Được tổ chức bởi VinTelligence — VinUni DS&AI Club

Tiêu chí

Mô tả

Điểm tối đa

lượng trực

Chất
quan hoá

Chiều sâu phân
tích

Biểu đồ có tiêu đề, nhãn trục, chú
thích đầy đủ; lựa chọn loại biểu đồ
phù hợp; thẩm mỹ trình bày rõ ràng
Bao phủ đầy đủ bốn cấp độ Descrip-
tive → Diagnostic → Predictive →
Prescriptive;
lập luận logic, có số
liệu cụ thể hỗ trợ

Insight kinh doanh Phát hiện có giá trị thực tiễn; đề
xuất hành động khả thi; liên kết rõ
ràng giữa dữ liệu và quyết định kinh
doanh
Góc nhìn độc đáo, không lặp lại các
phân tích hiển nhiên; mạch trình
bày coherent; kết nối nhiều bảng dữ
liệu một cách có chủ đích

Tính sáng tạo & kể
chuyện

15

25

15

5

Tổng tối đa

60 điểm

Chi tiết thang điểm từng tiêu chí:

Tiêu chí

Chất lượng
trực quan hoá
(15đ)

Chiều sâu
phân tích
(25đ)

Insight
kinh doanh
(15đ)

Tính sáng tạo
(5đ)

Mức
điểm

13–15đ

8–12đ

0–7đ

21–25đ

14–20đ

7–13đ

0–6đ

13–15đ

8–12đ

0–7đ

4–5đ

2–3đ

0–1đ

Mô tả

Tất cả biểu đồ đều đạt chuẩn, lựa chọn loại
biểu đồ tối ưu cho từng insight
Phần lớn biểu đồ đạt yêu cầu, một số thiếu
nhãn hoặc chú thích
Biểu đồ thiếu thông tin, khó đọc hoặc không
phù hợp với dữ liệu

Đạt cả bốn cấp độ Descriptive, Diagnostic,
Predictive, Prescriptive một cách nhất quán
Đạt ba cấp độ, cấp độ Prescriptive còn hời
hợt
Chủ yếu ở cấp Descriptive và Diagnostic

Chỉ mô tả bề mặt, thiếu phân tích

Đề xuất cụ thể, định lượng được, áp dụng
được ngay
Có đề xuất nhưng còn chung chung

Thiếu kết nối với bối cảnh kinh doanh

Góc nhìn độc đáo, kết hợp nhiều nguồn dữ
liệu, mạch trình bày thuyết phục
Có điểm sáng tạo nhưng chưa nhất quán

Phân tích dự đoán được, không có điểm nổi
bật

Phần 3 — Mô hình Dự báo Doanh thu (20 điểm)

Điểm Phần 3 được tính từ hai thành phần: hiệu suất mô hình trên Kaggle và chất lượng báo
cáo kỹ thuật.

14

DATATHON 2026 — The Gridbreakers

Được tổ chức bởi VinTelligence — VinUni DS&AI Club

Thành phần

Mô tả

Điểm tối đa

Hiệu suất mô hình

Báo cáo kỹ thuật

12

8

Dựa trên điểm MAE, RMSE,
R2 trên tập test (Kaggle leader-
board); xếp hạng tương đối so
với các đội khác trong cuộc thi
Chất lượng pipeline (feature en-
gineering, cross-validation, xử
lý leakage); giải thích mô hình
bằng SHAP / feature impor-
tance; tuân thủ các ràng buộc
đã nêu

Tổng tối đa

20 điểm

Thành phần

Hiệu suất mô hình
(12đ)

Báo cáo kỹ thuật
(8đ)

Mức
điểm

10–12đ

5–9đ

3–4đ

7–8đ

4–6đ

0–3đ

Mô tả

Xếp hạng top leaderboard; MAE và
RMSE thấp, R2 cao
Hiệu suất trung bình; mô hình hoạt động
nhưng chưa tối ưu
Bài nộp hợp lệ nhưng hiệu suất thấp; mức
điểm sàn

Pipeline rõ ràng, cross-validation đúng
chiều thời gian, giải thích mô hình cụ thể
bằng SHAP hoặc tương đương, tuân thủ
đầy đủ ràng buộc
Pipeline đủ dùng, giải thích còn định tính,
một số ràng buộc chưa được xử lý tường
minh
Thiếu giải thích, không kiểm soát leakage,
hoặc không thể tái lập kết quả

Điều kiện loại bài: Bài nộp sẽ bị loại toàn bộ Phần 3 nếu vi phạm bất kỳ ràng buộc
nào sau đây: (1) sử dụng Revenue/COGS từ tập test làm đặc trưng; (2) sử dụng dữ liệu
ngoài bộ dữ liệu được cung cấp; (3) không đính kèm mã nguồn hoặc kết quả không thể tái
lập.

15

DATATHON 2026 — The Gridbreakers

Được tổ chức bởi VinTelligence — VinUni DS&AI Club

Hướng dẫn Nộp bài

Checklist nộp bài

Mỗi đội cần hoàn thành và nộp đầy đủ các mục sau:

1. Nộp kết quả dự báo trên Kaggle

2. Link Kaggle: https://www.kaggle.com/competitions/datathon-2026-round-1

Đảm bảo file submission.csv có đúng số lượng dòng và giữ nguyên thứ tự như
sample_submission.csv. File không đúng định dạng sẽ bị từ chối bởi hệ thống
Kaggle.

3. Báo cáo (Report)

Viết báo cáo sử dụng template LaTeX của NeurIPS, có thể tải tại:

https://neurips.cc/Conferences/2025/CallForPapers

Yêu cầu báo cáo:

• Giới hạn tối đa 4 trang (không tính references và appendix)
• Bao gồm các nội dung:

– Trực quan hoá và phân tích dữ liệu (Phần 2)
– Phương pháp tiếp cận, pipeline mô hình và kết quả thực nghiệm (Phần 3)
• Đính kèm link GitHub repository của nhóm trong báo cáo (chứa toàn bộ mã

nguồn, notebook, và file submission)

Lưu ý: GitHub repository cần được đặt ở chế độ public hoặc cấp quyền truy cập
cho ban tổ chức trước deadline nộp bài. Repository nên có README.md mô tả cấu trúc
thư mục và hướng dẫn chạy lại kết quả.

4. Form nộp bài thi Vòng 1

Điền đầy đủ thông tin trong form nộp bài chính thức (link sẽ được cung cấp). Form yêu
cầu:

• Chọn đáp án đúng cho câu hỏi trắc nghiệm
• Upload file báo cáo (PDF)
• Link GitHub repository
• Link submission trên Kaggle
• Ảnh chụp thẻ sinh viên của tất cả thành viên trong đội
• Tickbox xác nhận: Nhóm thi cam kết có ít nhất 1 thành viên có thể tham gia

trực tiếp Vòng Chung kết vào ngày 23/05/2026 tại Đại học VinUni, Hà Nội

Quan trọng: Các đội không xác nhận khả năng tham gia trực tiếp Vòng Chung kết
hoặc không cung cấp đầy đủ ảnh thẻ học sinh sẽ không đủ điều kiện để được xét
vào vòng tiếp theo.

16


