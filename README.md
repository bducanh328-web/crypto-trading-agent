# crypto-trading-agent

Hệ thống long/short hợp đồng vĩnh cửu USDT trên Bitunix: **Lớp bối cảnh + BOS hai chiều**. Tách riêng khỏi
hệ thống cổ phiếu (vn-trading-agent). Vốn 100 triệu đồng, sụt giảm tối đa chấp nhận 15%.

**Mọi thông số là giá trị khởi điểm, CHƯA KIỂM ĐỊNH.** Không dùng để vào lệnh thật trước khi qua cổng kiểm định
(giai đoạn 2–3) và 3 tháng chạy giấy (giai đoạn 4).

## Giai đoạn 1 (hiện tại): dữ liệu + đèn chung

`cta den` chạy mỗi ngày lúc 07:05 giờ Việt Nam (sau khi nến 1D đóng), gửi Telegram:

| Đèn | BTC so MA200 (1D) | Cấu trúc swing BTC (1D) | Độ rộng: % rổ trên MA50 | Được làm |
| --- | --- | --- | --- | --- |
| XANH | Trên | BOS tăng gần nhất | ≥ 55% | Chỉ long |
| ĐỎ | Dưới | BOS giảm gần nhất | ≤ 40% | Chỉ short |
| VÀNG | Còn lại | | | Không mở lệnh mới |

- Đèn chỉ đổi khi điều kiện mới giữ 2 ngày liên tiếp.
- Rổ: 30 hợp đồng có giá trị giao dịch 30 ngày lớn nhất trên Bitunix; bỏ stablecoin, coin niêm yết dưới 180 ngày,
  hợp đồng không OPEN hoặc không cho giao dịch qua API. Lưu theo ngày ở `data/ro_coin.csv`.
- Swing: đỉnh/đáy so với 3 nến mỗi bên, chỉ dùng sau khi đủ 3 nến bên phải (không nhìn trước).
- Bộ lọc funding (trung bình 3 ngày, quy về mỗi 8 giờ): không long khi > 0,05%, không short khi < −0,03%.
- Lịch sử đèn: `data/den.csv`.

`cta lich-su BTCUSDT ETHUSDT --khung 1d,4h --tu 2020-01-01 --funding`: tải lịch sử dài từ data.binance.vision (tệp
tĩnh, không bị chặn vùng như fapi.binance.com) cho kiểm định giai đoạn 2.

## Cài đặt

```bash
pip install -e ".[dev]"
pytest -q
cta den               # in bản tin
cta den --telegram    # gửi Telegram
```

GitHub → Settings → Secrets and variables → Actions:

- Secrets: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (có thể dùng lại bot Telegram của hệ thống cổ phiếu).
- Biến `RUNNER` (tuỳ chọn): nếu Bitunix chặn máy GitHub đặt ở Mỹ, cài một máy tự host cho repo này rồi đặt
  `RUNNER` = nhãn của máy đó. Máy tự host của vn-trading-agent chỉ chạy cho repo đó, phải đăng ký thêm cho repo này.

## Nguồn dữ liệu và giới hạn đã biết

- Bitunix công khai: nến tối đa 200 cây mỗi lượt (bot lật trang), 10 lượt/giây. Tài liệu không nói rõ `quoteVol` /
  `baseVol` là đơn vị nào: bot tự suy từ dữ liệu (khối lượng coin × giá ≈ giá trị USDT).
- **Chưa chạy với API thật**: máy viết mã không ra được Internet. Lần chạy đầu trên GitHub cần xem log.
- Lịch sử nến 1D của rổ chỉ 200 ngày (đủ cho MA50); BTC 400 ngày (đủ cho MA200).
- Pháp lý: Nghị quyết 05/2025/NQ-CP yêu cầu giao dịch tài sản mã hoá qua tổ chức được cấp phép; cần xem lại trước
  giai đoạn 5 (tiền thật).

## Lộ trình

1. Dữ liệu + đèn chung (repo này, hiện tại)
2. Kiểm định trong mẫu 2020–2023 → cổng A
3. Ngoài mẫu 2024–10/2026 + 7 biến thể → cổng B
4. Chạy giấy 3 tháng, tín hiệu 4H 6 lần/ngày → cổng C
5. Tiền thật: bot đặt lệnh limit + stop qua API Bitunix, r = 0,25% cho 50 lệnh đầu
