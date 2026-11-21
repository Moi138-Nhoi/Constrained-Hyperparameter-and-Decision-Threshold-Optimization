# Đối chiếu yêu cầu giảng viên

Nguồn đối chiếu: `DE_BAI_TAP_LON_CA_NHAN_5_TUAN_GITHUB_PORTFOLIO.pdf` (6 trang, nhất là trang 4–6), `HUONG_DAN_PHAM_VI_DE_BAI_TAP_LON_GITHUB_PORTFOLIO.pdf` (5 trang) và tài liệu Chương 6 (29 trang), do người dùng cung cấp. Không đưa bản sao các tài liệu giảng viên vào repo.

| Yêu cầu | Bằng chứng trong bản hoàn thiện | Giới hạn |
|---|---|---|
| (1) Đặt vấn đề | Báo cáo trang 4; README Đặt vấn đề | Phát hiện trạng thái hiện tại, không dự báo tương lai |
| (2) Mô hình hóa | Trang 4–5; config.json; src/modeling.py | F₂ OOF, 6 siêu tham số + τ, FPR/size/latency; cap là giả định môn học |
| (3) Baseline + tối ưu | Trang 5, 7; 5 baseline; 40 Random Search + 40 TPE | Một ngân sách; không khẳng định ưu thế phổ quát |
| (4) Dữ liệu và thiết kế | Trang 6; data/README.md; manifest và split/fold CSV | UCI synthetic, CC BY 4.0; tên thương mại CPU và RAM không có trong log gốc |
| (5) Bảng/hình/khả thi/runtime | Trang 7–10; CSV/JSON; 2 đồ thị + 2 hình log thật | Runtime theo môi trường đã ghi; không có wall time riêng từng optimizer |
| (6) Thảo luận/sensitivity/thất bại | Trang 10–11; sensitivity.csv, stability.csv, ablation.csv | Ba seed thay đổi RF và cách chia fold, giữ τ; không phải ba lần HPO độc lập |
| (7) Kết luận chọn nghiệm | Trang 12; candidate_ranking.csv, refit_attempts.csv | Chọn theo development; test không hoàn toàn mù về lịch sử |
| Word/PDF 8–12 trang | bao_cao_do_an_ai4i.docx và PDF 12 trang | Word chỉnh sửa được; cần xuất lại/kiểm tra nếu sửa nội dung |
| 1 trang tóm tắt hoặc 5–8 slide | tom_tat_bao_ve_1_trang.docx và PDF 1 trang | Chưa xác nhận khả năng bảo vệ trực tiếp của sinh viên |
| Liên kết GitHub trong báo cáo | URL đầy đủ ở trang 2, 11 và tóm tắt | LMS do sinh viên nộp |
| Tái lập và kiểm chứng | requirements, unittest, verify_results, model joblib | Verifier đọc mô hình đã lưu; không coi là thí nghiệm test độc lập mới |
| Byte/hash nhất quán | .gitattributes; data/eol_migration.json; build_metadata.json | Hash chính xác theo byte LF, không bỏ qua lỗi |
| ZIP cùng commit | scripts/package_submission.py, git archive, inventory ngoài ZIP | Chỉ đóng gói tree đã commit sạch |
| Tiến độ 5 tuần | Giữ nguyên 6 commit cũ ngày 09/11/2026 | **Chưa chứng minh được tiến độ 5 tuần; không sửa ngày commit** |
| Công khai / About / topics | Kiểm tra và cập nhật qua GitHub tại thời điểm bàn giao | Trạng thái GitHub có thể đổi sau bàn giao |
| AI assistance | README và báo cáo trang 11 | Sinh viên chịu trách nhiệm kiểm chứng và giải thích |

Không tuyên bố đáp ứng toàn bộ rubric: lịch sử 5 tuần, test độc lập hoàn toàn, môi trường triển khai thực, nộp LMS và bảo vệ cá nhân vẫn là giới hạn.
