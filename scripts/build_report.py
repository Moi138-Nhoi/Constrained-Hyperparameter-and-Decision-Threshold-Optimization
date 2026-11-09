"""Build a 10-page Vietnamese PDF report and a one-page defense summary.

All numerical values come from one verified official run. Personal and LMS
fields remain visibly unfilled because they are absent from the dataset/logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
NAVY, BLUE, GRAY = "#183153", "#1267a8", "#475569"


def new_page(title: str, number: int, source_hash: str):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(0.08, 0.955, title, fontsize=17, fontweight="bold", color=NAVY)
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.938, 0.938], transform=fig.transFigure,
                              color=BLUE, linewidth=1.6))
    fig.text(0.08, 0.035, f"AI4I 2020 • Trang {number}/10 • run SHA-256 {source_hash[:12]}",
             fontsize=7.8, color=GRAY)
    return fig


def section(fig, y, heading, text=None, *, size=10.2):
    fig.text(0.085, y, heading, fontsize=12, fontweight="bold", color=BLUE)
    y -= 0.034
    if text:
        y = paragraph(fig, y, text, size=size)
    return y - 0.013


def paragraph(fig, y, text, *, size=9.8, width=91, line_step=0.025):
    for part in text.split("\n"):
        lines = textwrap.wrap(part, width=width, break_long_words=False,
                              break_on_hyphens=False) or [""]
        for line in lines:
            fig.text(0.09, y, line, fontsize=size, color=NAVY, va="top")
            y -= line_step
    return y - 0.012


def bullet(fig, y, text, *, size=9.7):
    return paragraph(fig, y, "• " + text, size=size, width=89)


def table(fig, y, headers, rows, *, height=0.22, widths=None, fontsize=8.2):
    ax = fig.add_axes([0.08, y - height, 0.84, height])
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="left",
                   colWidths=widths)
    tab.auto_set_font_size(False)
    tab.set_fontsize(fontsize)
    tab.scale(1, 1.37)
    for (row, _), cell in tab.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if row == 0:
            cell.set_facecolor("#dbeafe")
            cell.set_text_props(weight="bold", color=NAVY)
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
    return y - height - 0.022


def save_pdf(fig, pdf):
    pdf.savefig(fig, bbox_inches=None)
    plt.close(fig)


def build(results: Path, output: Path) -> dict:
    summary_path = results / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete":
        raise ValueError("A 10-page final report requires a feasible official full run")
    output.mkdir(parents=True, exist_ok=True)
    summary_hash = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    selected, test, cfg = summary["selected_final"], summary["test"], manifest["config"]
    baselines = pd.read_csv(results / "baselines.csv")
    trials = pd.read_csv(results / "trials.csv")
    sensitivity = pd.read_csv(results / "sensitivity.csv")
    stability = pd.read_csv(results / "stability.csv")
    ablation = pd.read_csv(results / "ablation.csv") if (results / "ablation.csv").exists() else None
    attempts = pd.read_csv(results / "refit_attempts.csv")
    source = "https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset"

    report_path = output / "bao_cao_do_an_ai4i.pdf"
    with PdfPages(report_path) as pdf:
        # 1. Cover and abstract
        fig = new_page("BÁO CÁO ĐỒ ÁN CÁ NHÂN", 1, summary_hash)
        fig.text(0.09, 0.86, "NHẬP MÔN TỐI ƯU HÓA", fontsize=13, weight="bold", color=BLUE)
        fig.text(0.09, 0.79, "Tối ưu siêu tham số và ngưỡng phân loại", fontsize=15, weight="bold", color=NAVY)
        fig.text(0.09, 0.75, "của Random Forest để phát hiện lỗi máy móc", fontsize=14, weight="bold", color=NAVY)
        fig.text(0.09, 0.71, "dưới ràng buộc cảnh báo sai và tài nguyên", fontsize=12, color=NAVY)
        y = 0.62
        y = section(fig, y, "Thông tin cần sinh viên điền")
        y = bullet(fig, y, "Họ tên: [CẦN ĐIỀN]; Mã số sinh viên: [CẦN ĐIỀN]")
        y = bullet(fig, y, "Giảng viên, lớp, đường dẫn nộp LMS: [CẦN ĐIỀN]")
        y = section(fig, y - 0.03, "Tóm tắt")
        y = paragraph(fig, y, "Dùng dữ liệu AI4I 2020 tổng hợp để phân loại lỗi ở quan sát hiện tại. "
                      "Tối ưu đồng thời sáu siêu tham số Random Forest và ngưỡng cảnh báo theo F₂ trên "
                      "dự đoán out-of-fold, ràng buộc FPR, kích thước tệp mô hình và tỷ lệ độ trễ. "
                      "So sánh Random Search và Optuna TPE cùng ngân sách 40 trial mỗi phương pháp; "
                      "kiểm tra nghiệm sau khi fit lại trước khi đánh giá test.")
        y = section(fig, y - 0.02, "Kết quả chính")
        y = paragraph(fig, y, f"Nghiệm cuối: {selected['method']} trial {selected['trial']}, "
                      f"ngưỡng {selected['threshold']:.4f}. F₂ OOF {selected['f2']:.4f}; "
                      f"F₂ test {test['f2']:.4f}; FPR test {test['fpr']:.4f}. "
                      f"Tỷ lệ độ trễ sau fit lại {test['latency_ratio']:.3f}× và "
                      f"kích thước {test['model_size_mb']:.3f} MB.")
        paragraph(fig, y - 0.03, "Báo cáo được tạo trực tiếp từ log chính thức; mọi số liệu được kiểm tra "
                  "bằng scripts/verify_results.py. Dataset mang tính tổng hợp, không chứng minh hiệu quả "
                  "triển khai trên máy thật.")
        save_pdf(fig, pdf)

        # 2. Problem and dataset
        fig = new_page("1. Đặt vấn đề và dữ liệu", 2, summary_hash)
        y = 0.89
        y = section(fig, y, "Bài toán thực tế", "Tại thời điểm quan sát, sáu thông số vận hành được dùng để quyết định có phát "
                    "cảnh báo lỗi máy hay không. False negative bỏ sót lỗi; false positive tạo cảnh báo sai. "
                    "F₂ ưu tiên recall nhưng cần giới hạn FPR để cảnh báo không quá dày.")
        y = section(fig, y, "Nguồn và bản chất dữ liệu", "UCI AI4I 2020 Predictive Maintenance Dataset, DOI "
                    "10.24432/C5HS5C, giấy phép CC BY 4.0. UCI mô tả đây là dữ liệu tổng hợp nhằm mô phỏng "
                    "dữ liệu bảo trì công nghiệp. Bài toán này là phát hiện nhãn Machine failure cùng hàng, "
                    "không phải dự báo hỏng hóc trong tương lai.")
        y = table(fig, y, ["Đặc điểm", "Giá trị"], [
            ["Số quan sát / cột", "10.000 / 14"],
            ["Nhãn lỗi / bình thường", "339 / 9.661"],
            ["Ô thiếu", "0"],
            ["Lệch nhãn tổng và OR cờ lỗi", "27; giữ nguyên nhãn tổng"],
            ["Split stratified", "8.000 development / 2.000 test"],
        ], height=0.25, widths=[0.45, 0.55])
        y = section(fig, y, "Kiểm soát rò rỉ thông tin")
        y = bullet(fig, y, "X chỉ gồm Type, hai nhiệt độ, tốc độ quay, mô-men xoắn, thời gian mòn dụng cụ.")
        y = bullet(fig, y, "Loại UDI, Product ID, TWF, HDF, PWF, OSF, RNF và chính Machine failure khỏi X.")
        y = bullet(fig, y, "One-hot Type được fit trong từng fold train. Không oversample trước split; không dùng SMOTE.")
        paragraph(fig, y - 0.01, "Nguồn: " + source, size=8.0, width=100)
        save_pdf(fig, pdf)

        # 3. Formulation
        fig = new_page("2. Mô hình hóa tối ưu", 3, summary_hash)
        y = 0.89
        y = section(fig, y, "Biến quyết định", "λ = (n_estimators, max_depth, min_samples_split, "
                    "min_samples_leaf, max_features, class_weight) thuộc miền rời rạc trong config.json; "
                    "τ là ngưỡng xác suất trong [0,1]. Với mỗi λ, Random Forest được fit riêng trong ba fold.")
        y = section(fig, y, "Hàm mục tiêu và miền khả thi")
        y = paragraph(fig, y, "Tối đa F₂(λ,τ) = 5TP/(5TP + 4FN + FP) trên toàn bộ xác suất OOF của tập "
                      "development. Ràng buộc: FPR = FP/(FP + TN) ≤ 0,05; model size tối đa theo fold "
                      "≤ 20 MB; trung vị tỷ lệ latency ghép cặp ứng viên / RF mặc định ≤ 1,25.")
        y = paragraph(fig, y, "Các giới hạn 5%, 20 MB và 1,25× là giả định bài tập, không phải chuẩn công nghiệp. "
                      "Nếu không tìm được nghiệm phù hợp, chương trình ghi trạng thái vô nghiệm và không nới "
                      "giới hạn hay báo một kết quả test giả là khả thi.")
        y = section(fig, y, "Cách chọn ngưỡng", "Với xác suất OOF cố định, xét tất cả mức xác suất phân biệt "
                    "và trường hợp không báo động khi có thể. Trong tập đạt FPR, chọn F₂ cao nhất; khi hòa "
                    "chọn FPR thấp hơn, rồi ngưỡng cao hơn. Hai ngưỡng FPR 1% và 2% chỉ để phân tích độ nhạy.")
        y = section(fig, y, "Đánh đổi", "Một ngưỡng thấp thường tăng recall nhưng cũng tăng false positives. "
                    "Nhiều cây hoặc cây sâu có thể tăng F₂ nhưng tăng thời gian suy luận và kích thước. "
                    "Không đánh giá nghiệm chỉ bằng một metric duy nhất.")
        save_pdf(fig, pdf)

        # 4. Protocol
        fig = new_page("3. Phương pháp và quy trình", 4, summary_hash)
        y = 0.89
        y = section(fig, y, "Dữ liệu và baseline", "Khóa test stratified 20% với seed 42. Trên 80% còn lại, "
                    "dùng StratifiedKFold 3 fold xáo trộn cố định. Baseline gồm Dummy, RF mặc định và RF "
                    "balanced tại ngưỡng 0,5; thêm hai baseline giữ RF cố định nhưng chọn ngưỡng trên OOF.")
        y = section(fig, y, "Hai phương pháp tìm kiếm", "Random Search lấy đều từ cùng các lựa chọn rời rạc; "
                    "Optuna TPE dùng lịch sử trial sau 10 lần khởi tạo. Mỗi phương pháp có 40 trial, cùng "
                    "folds, tính năng đầu vào, số luồng và phép đo. Cấu hình không đạt ràng buộc nhận trạng "
                    "thái infeasible, không thể được chọn.")
        y = section(fig, y, "Latency ghép cặp", "Cùng batch 256 hàng, một RF worker và một luồng thư viện "
                    "số. Sau ba warm-up, đo 15 vòng mỗi fold với thứ tự ABBA rồi BAAB; mỗi vòng tính trung "
                    "bình hai lượt ứng viên và hai lượt baseline, sau đó lấy tỷ lệ. Quyết định dùng trung vị "
                    "45 tỷ lệ; q25, q75, MAD và toàn bộ vòng đo được lưu.")
        y = section(fig, y, "Chọn nghiệm và test", "Xếp hạng nghiệm OOF hợp lệ theo F₂, FPR, kích thước, "
                    "latency, method và trial. Fit lại trên 8.000 hàng development theo thứ tự này, kiểm "
                    "tra size và latency trước test; nếu ứng viên đầu trượt thì thử ứng viên dự phòng. "
                    "Chỉ nghiệm đầu đạt được đánh giá test.")
        save_pdf(fig, pdf)

        # 5. Reproducibility
        fig = new_page("4. Thiết kế thực nghiệm và tái lập", 5, summary_hash)
        y = 0.83
        runtime = manifest["runtime"]
        y = section(fig, y, "Môi trường chạy")
        y = table(fig, y, ["Mục", "Giá trị"], [
            ["Python / hệ điều hành", f"{runtime['python']} / {runtime['platform']}"],
            ["CPU", runtime["processor"]],
            ["NumPy / pandas", f"{runtime['packages']['numpy']} / {runtime['packages']['pandas']}"],
            ["scikit-learn / Optuna", f"{runtime['packages']['scikit-learn']} / {runtime['packages']['optuna']}"],
            ["Seed / CV", f"{cfg['seed']} / {cfg['folds']} fold"],
            ["Thời gian full run", f"{summary['wall_seconds'] / 60:.1f} phút"],
        ], height=0.28, widths=[0.35, 0.65], fontsize=7.7)
        y = section(fig, y, "Dấu vết tái lập", "manifest.json ghi hash CSV, cấu hình, phiên bản, CPU và "
                    "wall time. split_assignments.csv ghi development/test; fold_assignments.csv ghi "
                    "fold của 8.000 hàng development. trials.csv lưu từng cấu hình, threshold, confusion "
                    "matrix, metric, size, latency và trạng thái. refit_attempts.csv lưu thứ tự dự phòng.")
        y = section(fig, y, "Lệnh")
        y = paragraph(fig, y, "uv venv .venv  →  uv pip install -r requirements.txt  →  "
                      ".venv\\Scripts\\python.exe run.py --output-dir results/full  →  "
                      ".venv\\Scripts\\python.exe scripts\\verify_results.py", width=82)
        y = paragraph(fig, y, "Thư mục output phải trống để tránh trộn nhiều lần chạy. Pilot có nhãn "
                      "riêng và không đánh giá test. Các bản vẽ trong báo cáo lấy trực tiếp từ full run.")
        save_pdf(fig, pdf)

        # 6. Baselines and search
        fig = new_page("5. Baseline và tìm kiếm", 6, summary_hash)
        y = 0.89
        y = section(fig, y, "So sánh trên development OOF")
        rows = [[str(row["method"]), f"{row['f2']:.4f}", f"{row['fpr']:.4f}",
                 f"{row['threshold']:.4f}"] for _, row in baselines.iterrows()]
        for method in ("random_search", "optuna_tpe"):
            feasible = trials[(trials["method"] == method) & trials["feasible"]]
            if len(feasible):
                row = feasible.sort_values(["f2", "fpr"], ascending=[False, True]).iloc[0]
                rows.append([f"Best {method} #{int(row['trial'])}", f"{row['f2']:.4f}",
                             f"{row['fpr']:.4f}", f"{row['threshold']:.4f}"])
            else:
                rows.append([f"Best {method}", "Không có nghiệm", "—", "—"])
        y = table(fig, y, ["Ứng viên", "F₂", "FPR", "τ"], rows, height=0.32,
                  widths=[0.49, 0.17, 0.17, 0.17], fontsize=8.1)
        y = section(fig, y, "Tính khả thi và chi phí", f"Random Search có "
                    f"{summary['search_counts']['random_search']}/{cfg['trials_per_method']} trial hợp lệ; "
                    f"TPE có {summary['search_counts']['optuna_tpe']}/{cfg['trials_per_method']}. "
                    "Số liệu này phản ánh riêng một lần chạy, không chứng minh ưu thế thuật toán tổng quát.")
        y = paragraph(fig, y, "Baseline RF tối ưu ngưỡng OOF cho phép thấy phần cải thiện đến từ τ dù không "
                      "thay λ. Vì vậy, không gán toàn bộ mức tăng F₂ so với ngưỡng 0,5 cho HPO.")
        save_pdf(fig, pdf)

        # 7. Figures
        fig = new_page("6. Diễn biến tối ưu và đánh đổi", 7, summary_hash)
        top = fig.add_axes([0.09, 0.51, 0.82, 0.38])
        top.imshow(plt.imread(results / "figures" / "optimization_history.png"))
        top.axis("off")
        bottom = fig.add_axes([0.09, 0.10, 0.82, 0.36])
        bottom.imshow(plt.imread(results / "figures" / "f2_fpr.png"))
        bottom.axis("off")
        save_pdf(fig, pdf)

        # 8. Decision and test
        fig = new_page("7. Nghiệm cuối và đánh giá test", 8, summary_hash)
        y = 0.89
        y = section(fig, y, "Quyết định sau khi fit lại", f"Ứng viên đứng đầu OOF: "
                    f"{summary['top_development_candidate']['method']} trial "
                    f"{summary['top_development_candidate']['trial']}. "
                    f"Số lượt fit lại đã thử: {len(attempts)}. "
                    f"Ứng viên cuối: {selected['method']} trial {selected['trial']}; "
                    f"dùng dự phòng: {summary['fallback_used']}. Không dùng metric test để chọn dự phòng.")
        y = paragraph(fig, y, f"Thông số: {selected['params_json']}; τ = {selected['threshold']:.6f}. "
                      f"F₂ OOF {selected['f2']:.4f}, FPR OOF {selected['fpr']:.4f}.", width=85)
        y = table(fig, y, ["TN", "FP", "FN", "TP"], [[str(test[k]) for k in ("tn", "fp", "fn", "tp")]],
                  height=0.12, widths=[0.25] * 4)
        y = table(fig, y, ["Metric test", "Giá trị"], [
            ["F₂ / recall / precision", f"{test['f2']:.4f} / {test['recall']:.4f} / {test['precision']:.4f}"],
            ["FPR / average precision", f"{test['fpr']:.4f} / {test['average_precision']:.4f}"],
            ["Accuracy", f"{test['accuracy']:.4f}"],
            ["Size cuối / giới hạn", f"{test['model_size_mb']:.3f} / {cfg['max_model_mb']:.1f} MB"],
            ["Latency ratio / giới hạn", f"{test['latency_ratio']:.3f} / {cfg['max_latency_ratio']:.2f}×"],
            ["Latency ratio q25–q75", f"{test['latency_ratio_q25']:.3f}–{test['latency_ratio_q75']:.3f}×"],
        ], height=0.29, widths=[0.48, 0.52])
        paragraph(fig, y, "FPR trên test là quan sát giữ lại, không phải mục tiêu được tối ưu trực tiếp. "
                  "Tỷ lệ latency được đo trước khi đọc test và có thể thay đổi trên máy khác.")
        save_pdf(fig, pdf)

        # 9. Diagnostics
        fig = new_page("8. Độ nhạy, ổn định và ablation", 9, summary_hash)
        y = 0.89
        selected_sens = sensitivity[(sensitivity["method"] == selected["method"]) &
                                    (sensitivity["trial"] == selected["trial"])]
        y = section(fig, y, "Độ nhạy theo giới hạn FPR")
        y = table(fig, y, ["FPR cap", "τ", "F₂ OOF", "FPR thực"], [
            [f"{row['fpr_cap']:.0%}", f"{row['threshold']:.4f}",
             f"{row['f2']:.4f}", f"{row['fpr']:.4f}"] for _, row in selected_sens.iterrows()
        ], height=0.19, widths=[0.25] * 4)
        y = section(fig, y, "Ổn định với ngưỡng và siêu tham số cố định")
        y = table(fig, y, ["Seed", "F₂ OOF", "Recall", "FPR"], [
            [str(int(row["seed"])), f"{row['f2']:.4f}", f"{row['recall']:.4f}",
             f"{row['fpr']:.4f}"] for _, row in stability.iterrows()
        ], height=0.19, widths=[0.25] * 4)
        if ablation is not None:
            y = section(fig, y, "Ablation đặc trưng tại quan sát hiện tại")
            y = table(fig, y, ["Engineered", "F₂ OOF", "FPR", "Latency ratio"], [
                [str(bool(row["engineered"])), f"{row['f2']:.4f}", f"{row['fpr']:.4f}",
                 f"{row['latency_ratio']:.3f}×"] for _, row in ablation.iterrows()
            ], height=0.16, widths=[0.25] * 4)
            y = paragraph(fig, y, "Ablation không tham gia chọn model cuối hoặc đánh giá test; "
                          "đặc trưng kỹ thuật chỉ dùng đại lượng có sẵn ở thời điểm dự đoán.")
        save_pdf(fig, pdf)

        # 10. Discussion
        fig = new_page("9. Thảo luận, hạn chế và kết luận", 10, summary_hash)
        y = 0.89
        y = section(fig, y, "Hạn chế")
        for item in [
            "AI4I là dữ liệu tổng hợp; không chứng minh hiệu quả trên máy móc thực hoặc khả năng dự báo lỗi tương lai.",
            "Split phân tầng ngẫu nhiên không đánh giá drift theo thời gian hay máy; test có ít ca lỗi nên metric vẫn biến động.",
            "Một pilot của bản cũ từng xem test trước lần chạy chính thức này. Không dùng số test đó để chọn tham số, nhưng test không còn hoàn toàn mù.",
            "Latency ghép cặp giảm nhiễu thứ tự/tải, vẫn phụ thuộc phần cứng; 20 MB là kích thước tệp, không phải RAM.",
            "Một split và một ngân sách 40 + 40 chưa chứng minh phương pháp tìm kiếm nào luôn tốt hơn.",
        ]:
            y = bullet(fig, y, item, size=9.3)
        y = section(fig, y - 0.01, "Kết luận")
        y = paragraph(fig, y, f"Theo protocol đã khóa, nghiệm cuối đạt F₂ test {test['f2']:.4f}, "
                      f"FPR {test['fpr']:.4f}, size {test['model_size_mb']:.3f} MB và tỷ lệ latency "
                      f"{test['latency_ratio']:.3f}×. Các ràng buộc bài tập được kiểm tra trước test; "
                      "kết luận giới hạn trong dataset tổng hợp và máy đo hiện tại.")
        y = section(fig, y, "Nguồn, AI và việc cần điền")
        y = paragraph(fig, y, "UCI AI4I 2020, DOI 10.24432/C5HS5C, CC BY 4.0. "
                      "OpenAI Codex hỗ trợ mã, tài liệu và kiểm tra; sinh viên tự chịu trách nhiệm tái lập "
                      "và giải thích. Họ tên, MSSV, giảng viên và LMS: [CẦN ĐIỀN].", size=9.2)
        paragraph(fig, y, "Repository: https://github.com/Moi138-Nhoi/Constrained-Hyperparameter-and-Decision-Threshold-Optimization",
                  size=8.2, width=105)
        save_pdf(fig, pdf)

    # A one-page alternative to the required 5–8 defense slides.
    one_page = output / "tom_tat_bao_ve_1_trang.pdf"
    with PdfPages(one_page) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.text(0.05, 0.92, "AI4I 2020 • TỐI ƯU RF VÀ NGƯỠNG CẢNH BÁO", fontsize=17,
                 weight="bold", color=NAVY)
        fig.text(0.05, 0.87, "Phát hiện lỗi ở quan sát hiện tại • dữ liệu tổng hợp UCI • seed 42",
                 fontsize=10.5, color=GRAY)
        ax = fig.add_axes([0.05, 0.30, 0.52, 0.51])
        ax.imshow(plt.imread(results / "figures" / "optimization_history.png"))
        ax.axis("off")
        box = [
            f"OOF F₂: {selected['f2']:.4f}",
            f"Test F₂: {test['f2']:.4f}",
            f"Test recall / precision: {test['recall']:.3f} / {test['precision']:.3f}",
            f"Test FPR / PR-AUC: {test['fpr']:.3f} / {test['average_precision']:.3f}",
            f"TN / FP / FN / TP: {test['tn']} / {test['fp']} / {test['fn']} / {test['tp']}",
            f"Size: {test['model_size_mb']:.3f} / {cfg['max_model_mb']:.0f} MB",
            f"Latency: {test['latency_ratio']:.3f} / {cfg['max_latency_ratio']:.2f}×",
            f"Fallback: {summary['fallback_used']}",
        ]
        for i, line in enumerate(box):
            fig.text(0.59, 0.80 - i * 0.064, line, fontsize=10.5, color=NAVY)
        fig.text(0.05, 0.17, "Phương pháp: 20% test khóa trước; 3-fold OOF; 40 Random + 40 TPE; "
                 "đo latency ghép cặp; kiểm tra fit lại trước test.", fontsize=9.3, color=NAVY)
        fig.text(0.05, 0.11, "Giới hạn: dữ liệu tổng hợp; pilot cũ từng xem test; timing phụ thuộc máy. "
                 "Thông tin cá nhân/LMS: [CẦN ĐIỀN].", fontsize=9.1, color=GRAY)
        fig.text(0.05, 0.05, f"Nguồn: UCI DOI 10.24432/C5HS5C • run SHA-256 {summary_hash[:12]}",
                 fontsize=8, color=GRAY)
        pdf.savefig(fig)
        plt.close(fig)

    source_results = str(results.resolve().relative_to(ROOT)) if results.resolve().is_relative_to(ROOT) else str(results)
    metadata = {"source_summary_sha256": summary_hash, "source_results": source_results,
                "report_pdf": report_path.name, "report_pages": 10,
                "defense_summary_pdf": one_page.name, "defense_summary_pages": 1,
                "placeholders": ["student_name", "student_id", "lecturer", "class", "LMS_link"]}
    (output / "build_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/full")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "report")
    args = parser.parse_args()
    print(json.dumps(build(args.results_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
