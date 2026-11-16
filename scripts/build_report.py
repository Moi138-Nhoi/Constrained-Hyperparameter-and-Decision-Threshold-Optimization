"""Generate editable DOCX reports from official logs, then seal exported PDFs.

python scripts/build_report.py
powershell -NoProfile -File scripts/export_report.ps1
python scripts/build_report.py --finalize-pdfs

PDF export may also use LibreOffice. Inspect the actual export before sealing.
This script never trains models or selects a new decision.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
REPO = 'https://github.com/Moi138-Nhoi/Constrained-Hyperparameter-and-Decision-Threshold-Optimization'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def para(doc, text='', size=None, bold=False, center=False):
    p = doc.add_paragraph()
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(str(text)); r.bold = bold
    if size:
        r.font.size = Pt(size)
    return p


def heading(doc, text, level=1):
    return doc.add_heading(text, level)


def page(doc, title):
    doc.add_page_break(); heading(doc, title)


def table(doc, headers, rows, size=10):
    t = doc.add_table(rows=1, cols=len(headers)); t.style = 'Table Grid'
    for c, text in zip(t.rows[0].cells, headers):
        c.text = str(text)
        c._tc.get_or_add_tcPr().append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="E8EEF6"/>'))
    for row in rows:
        for c, text in zip(t.add_row().cells, row):
            c.text = str(text)
    for i, row in enumerate(t.rows):
        row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(3)
                p.paragraph_format.space_before = Pt(3)
                p.paragraph_format.line_spacing = 1.05
                for r in p.runs:
                    r.font.size = Pt(size); r.bold = i == 0
    para(doc, '', size=3).paragraph_format.space_after = Pt(0)
    return t


def picture(doc, path, caption, width=16):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(3)
    p.add_run().add_picture(str(path), width=Cm(width))
    para(doc, caption, size=10, center=True)


def new_doc(brief=False):
    d = Document(); s = d.sections[0]
    s.page_width, s.page_height = Cm(21), Cm(29.7)
    s.top_margin, s.bottom_margin = Cm(1.8), Cm(1.7)
    s.left_margin, s.right_margin = Cm(2.3), Cm(2.1)
    s.header_distance = Cm(.7); s.footer_distance = Cm(.7)
    st = d.styles['Normal']; st.font.name = 'Times New Roman'
    st.font.size = Pt(11 if brief else 12)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.12
    for name, size in [('Title',19),('Heading 1',17),('Heading 2',13)]:
        st = d.styles[name]; st.font.name = 'Times New Roman'; st.font.size = Pt(size)
        st.font.color.rgb = RGBColor.from_string('17365D')
        st.paragraph_format.space_after = Pt(8); st.paragraph_format.space_before = Pt(5)
    s.different_first_page_header_footer = not brief
    p=s.header.paragraphs[0]
    p.add_run('NHẬP MÔN TỐI ƯU HÓA  |  RANDOM FOREST • AI4I 2020').font.size=Pt(8)
    p=s.footer.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.add_run('Huỳnh Trí Minh  •  ').font.size=Pt(9)
    fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),'PAGE'); p._p.append(fld)
    d.core_properties.author='Huỳnh Trí Minh'
    d.core_properties.title='Tối ưu Random Forest phát hiện lỗi máy'
    return d


def cover(d, output, identity):
    # Original decorative frame/logo behind editable cover text.
    inline=d.sections[0].first_page_header.paragraphs[0].add_run().add_picture(
        str(output/'assets/cover_motif.png'),width=Cm(21),height=Cm(29.7))._inline
    anchor=OxmlElement('wp:anchor')
    for key,value in {'distT':'0','distB':'0','distL':'0','distR':'0','simplePos':'0','relativeHeight':'0','behindDoc':'1','locked':'0','layoutInCell':'1','allowOverlap':'1'}.items():
        anchor.set(key,value)
    anchor.append(parse_xml('<wp:simplePos xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" x="0" y="0"/>'))
    for axis in ['H','V']:
        pos=OxmlElement('wp:position'+axis); pos.set('relativeFrom','page')
        off=OxmlElement('wp:posOffset'); off.text='0'; pos.append(off); anchor.append(pos)
    for child in list(inline): anchor.append(child)
    anchor.insert(4,OxmlElement('wp:wrapNone'))
    inline.getparent().replace(inline,anchor)
    def line(text,size,before=0,after=0,bold=True,color=None):
        p=para(d,text,size,bold,True); pf=p.paragraph_format
        pf.space_before=Pt(before); pf.space_after=Pt(after); pf.line_spacing=1
        if color: p.runs[0].font.color.rgb=RGBColor.from_string(color)
    line(identity['university'],15,13,7); line(identity['faculty'],14)
    line('BÁO CÁO ĐỒ ÁN MÔN HỌC',14,165,12)
    line('NHẬP MÔN TỐI ƯU HÓA',17,0,12,color='295398')
    line('NGÀNH: '+identity['major'],12,0,24,False)
    line('Đề tài',13,0,10)
    line('TỐI ƯU RANDOM FOREST',18,0,7,color='BB1725')
    line('PHÁT HIỆN LỖI MÁY',18,0,50,color='BB1725')
    line('SVTH: '+identity['student'],13,0,21)
    line('MSSV: '+identity['student_id'],13,0,21)
    line('GVHD: '+identity['lecturer'],13,0,58)
    line('THÀNH PHỐ HỒ CHÍ MINH – NĂM 2026',11)


def build(results,output):
    s=json.loads((results/'summary.json').read_text(encoding='utf-8'))
    m=json.loads((results/'manifest.json').read_text(encoding='utf-8'))
    if s['status']!='complete' or m['run_type']!='full': raise ValueError('A complete official run is required')
    cfg=m['config']; selected=s['selected_final']; test=s['test']; params=json.loads(selected['params_json'])
    b=pd.read_csv(results/'baselines.csv'); t=pd.read_csv(results/'trials.csv')
    sens=pd.read_csv(results/'sensitivity.csv'); stab=pd.read_csv(results/'stability.csv')
    abl=pd.read_csv(results/'ablation.csv')
    identity=json.loads((output/'identity.json').read_text(encoding='utf-8')); assets=output/'assets'
    d=new_doc(); cover(d,output,identity)

    page(d,'MỤC LỤC VÀ TÓM TẮT')
    table(d,['Nội dung','Trang'],[
        ['1. Đặt vấn đề','3'],['2. Mô hình hóa bài toán tối ưu','3–4'],
        ['3. Baseline và phương pháp tối ưu','4'],['4. Dữ liệu và thiết kế thực nghiệm','5'],
        ['5. Kết quả, hình và runtime/độ ổn định','6–9'],
        ['6. Trade-off, sensitivity, thất bại và hạn chế','9–10'],
        ['7. Kết luận và tái lập','11'],['Tài liệu tham khảo và đối chiếu yêu cầu','12']],11)
    heading(d,'Tóm tắt',2)
    para(d,'Đồ án xét phát hiện lỗi máy tại thời điểm quan sát trên AI4I 2020. Mục tiêu là tối đa F₂ trên dự đoán ngoài fold (OOF), đồng thời giới hạn cảnh báo sai, kích thước lưu và độ trễ suy luận. Năm baseline giúp tách hiệu quả chỉnh ngưỡng khỏi hiệu quả thay đổi siêu tham số. Random Search và Optuna TPE mỗi phương pháp có 40 trial trên cùng ba fold.')
    para(d,f'Nghiệm chốt theo development là {selected["method"]}, trial {selected["trial"]} (đánh số từ 0), ngưỡng {selected["threshold"]:.6f}; F₂ OOF {selected["f2"]:.4f}. Sau refit hợp lệ, test cho F₂ {test["f2"]:.4f}, recall {test["recall"]:.4f}, FPR {test["fpr"]:.4f}. Các số liệu được đọc từ results/full, không nhập số minh họa.')
    heading(d,'Nguồn và phạm vi của bản hoàn thiện',2)
    para(d,'Bản này giữ nguyên thực nghiệm ngày 09/11/2026, mô hình đã lưu và các hình chạy thật trong ZIP; bổ sung phân tích, Word chỉnh sửa được và kiểm chứng byte/hash. Chuẩn hóa CRLF thành LF không đổi giá trị CSV hay quyết định mô hình. Ngày hoàn thiện tài liệu không được coi là ngày chạy lại thí nghiệm.')
    para(d,'Pilot của triển khai cũ từng xem cùng tập test. Vì vậy test không hoàn toàn mù về lịch sử; quy trình chọn cấu hình hiện tại chỉ dùng development. Kiểm tra lại mô hình đã lưu trên test là kiểm tra tính nhất quán, không phải một đánh giá độc lập mới.',bold=True)
    para(d,'Kho mã nguồn công khai (URL đầy đủ):',size=10); para(d,REPO,size=9)

    page(d,'1. ĐẶT VẤN ĐỀ')
    para(d,'Trong giám sát máy, bỏ sót trạng thái lỗi có thể làm chậm xử lý, trong khi quá nhiều cảnh báo sai gây tốn công kiểm tra. Đầu vào là các phép đo đã có tại thời điểm quan sát; đầu ra là cảnh báo Machine failure = 1. Đây là phân loại trạng thái hiện tại, chưa phải dự báo máy sẽ hỏng sau một khoảng thời gian.')
    para(d,'Câu hỏi tối ưu: với cùng dữ liệu và ngân sách thử nghiệm, chọn siêu tham số Random Forest và ngưỡng nào để tăng phát hiện lỗi trong giới hạn FPR và tài nguyên? Accuracy riêng lẻ dễ gây hiểu nhầm vì lớp lỗi chỉ chiếm 3,39%; dự đoán luôn bình thường vẫn đạt accuracy 96,61% trên toàn bộ dữ liệu.')
    heading(d,'2. MÔ HÌNH HÓA BÀI TOÁN TỐI ƯU')
    heading(d,'2.1. Biến quyết định và hàm mục tiêu',2)
    para(d,'x = (λ, τ), với λ = (n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, class_weight) và τ ∈ [0, 1]. Với xác suất pᵢ từ mô hình không học hàng i, dự đoán ŷᵢ = 1[pᵢ ≥ τ]. Trọng số của từng cây là kết quả huấn luyện, không phải biến tìm kiếm bên ngoài.')
    para(d,'max F₂(λ, τ) = 5TP / (5TP + 4FN + FP)',size=14,bold=True,center=True)
    para(d,'TP, FP, FN, TN được cộng trên 8.000 dự đoán OOF. F₂ đặt trọng số β² = 4 cho recall trong trung bình điều hòa với precision; không có nghĩa chi phí tiền tệ của FN luôn gấp bốn FP. Average precision (AP) được báo thêm như tóm tắt PR-AUC không nội suy.')
    heading(d,'2.2. Ràng buộc và giả định',2)
    table(d,['Đại lượng','Điều kiện khả thi'],[
        ['FPR OOF = FP/(FP+TN)','≤ 0,05'],['Size mô hình fold lớn nhất','≤ 20 MB thập phân; joblib không nén'],
        ['Độ trễ tương đối','Median 45 tỷ số candidate/reference ≤ 1,25'],
        ['Sau refit toàn development','Kiểm tra lại size và latency trước test']],10.5)
    para(d,'Các giới hạn 5%, 20 MB và 1,25× là giả định môn học, chưa được nhà máy xác nhận. FPR development không bảo đảm FPR tương lai. Biến rời rạc, objective không trơn và latency nhiễu; bài làm không dùng gradient và không tuyên bố tối ưu toàn cục.')

    page(d,'2–3. MIỀN TÌM KIẾM VÀ PHƯƠNG PHÁP')
    table(d,['Biến','Miền dùng chung cho hai phương pháp'],[
        [k,', '.join('None' if v is None else str(v) for v in values)] for k,values in cfg['search_space'].items()],10.5)
    para(d,'Miền có 4 × 5 × 3 × 3 × 3 × 3 = 1.620 tổ hợp λ. None ở max_depth là không giới hạn độ sâu; None ở class_weight là không tái cân bằng. Miền hữu hạn giới hạn chi phí; cây sâu/nhiều cây có thể tăng khả năng biểu diễn nhưng tăng runtime và size.')
    heading(d,'3.1. Nguyên lý và năm baseline',2)
    para(d,'Random Forest học các cây trên mẫu bootstrap và chọn tập con đặc trưng tại mỗi phép chia; lấy trung bình xác suất của các cây. Pipeline học one-hot Type trong từng fold. Ba baseline cố định: Dummy lớp phổ biến nhất, RF mặc định τ=0,5, RF class_weight=balanced τ=0,5. Hai baseline bổ sung chọn τ từ OOF cho RF mặc định và RF balanced.')
    heading(d,'3.2. Random Search và Optuna TPE',2)
    para(d,'Random Search lấy mẫu đều từng biến phân loại, có hoàn lại, bằng NumPy seed 42. TPE dùng cùng miền và seed; 10 trial đầu khởi tạo ngẫu nhiên, sau đó ước lượng phân bố cấu hình tốt/kém từ lịch sử để đề xuất vùng triển vọng. Đầu vào trial là λ; đầu ra objective là F₂ khả thi, hoặc −1 nếu vi phạm. Đây là phạt bất khả thi, không phải chứng nhận tối ưu có ràng buộc.')
    para(d,'Mỗi phương pháp đủ 40 trial, không pruning, cùng ba fold và giao thức đo. Ngân sách bằng nhau theo trial, không theo giây. Với mỗi λ, duyệt các mức xác suất OOF phân biệt và ngưỡng không cảnh báo nếu tồn tại; chọn F₂ cao nhất dưới FPR cap, hòa thì ưu tiên FPR thấp và ngưỡng cao.')
    heading(d,'3.3. Khóa quyết định trước test',2)
    para(d,'Xếp nghiệm OOF khả thi (gồm baseline RF) theo F₂ giảm, FPR tăng, size tăng, latency ratio tăng, rồi method/trial. Khử trùng λ–τ, refit lần lượt theo thứ tự đã khóa. Nếu refit vi phạm size/latency thì thử nghiệm kế tiếp; nếu không có nghiệm hợp lệ thì dừng, không đánh giá test. Chỉ refit khả thi đầu tiên được báo kết quả test.')

    page(d,'4. DỮ LIỆU VÀ THIẾT KẾ THỰC NGHIỆM')
    heading(d,'4.1. Nguồn, quyền sử dụng và preprocessing',2)
    para(d,'AI4I 2020 Predictive Maintenance Dataset, UCI [1], DOI 10.24432/C5HS5C, giấy phép CC BY 4.0. Đây là dữ liệu tổng hợp mô phỏng công nghiệp, không phải dữ liệu đo từ nhà máy trong đồ án. CSV có 10.000 hàng, 14 cột, 339 nhãn lỗi, 9.661 nhãn bình thường; không ô thiếu hay hàng trùng hoàn toàn. Định danh trong CSV là UDI.')
    para(d,'Sáu đầu vào: Type; Air temperature [K]; Process temperature [K]; Rotational speed [rpm]; Torque [Nm]; Tool wear [min]. Loại UDI, Product ID, nhãn và TWF/HDF/PWF/OSF/RNF khỏi X. Có 27 hàng nhãn khác OR của năm cờ; giữ nhãn gốc. One-hot Type với handle_unknown=ignore; số giữ nguyên đơn vị, không SMOTE, không chuẩn hóa bắt buộc cho cây.')
    heading(d,'4.2. Split, fold và seed',2)
    table(d,['Thành phần','Quy định thực nghiệm'],[
        ['Development / test','8.000 / 2.000 hàng; 271 / 68 nhãn lỗi'],
        ['Chia dữ liệu','train_test_split stratify, test_size=0,2, seed=42'],
        ['Đánh giá development','StratifiedKFold(3, shuffle=True, random_state=42)'],
        ['Ngân sách','5 baseline + 40 Random Search + 40 TPE'],
        ['Độ ổn định bổ sung','Seed mô hình 42,43,44; giữ fold và τ đã chốt']],10.5)
    para(d,'split_assignments.csv và fold_assignments.csv lưu từng chỉ số hàng. Chia tuần tự 70/15/15 chỉ là chẩn đoán: validation và test không có HDF; không dùng cho kết quả chính. Chia ngẫu nhiên chưa đo tổng quát hóa theo thời gian hoặc theo máy.')
    heading(d,'4.3. Môi trường và đo runtime',2)
    para(d,f'Log gốc: {m["runtime"]["platform"]}; CPU: {m["runtime"]["processor"]}; Python {m["runtime"]["python"]}. RF n_jobs=1, threadpool_limit=1. Log không có tên thương mại CPU/RAM nên không suy đoán. Phiên bản thư viện khóa trong requirements.txt và manifest.json.')
    para(d,'Đo predict_proba toàn pipeline trên cùng batch 256 hàng development/validation: 3 warm-up, 15 vòng ghép cặp, hai lần/model/vòng; đảo ABBA và BAAB. Gộp 45 tỷ số từ ba fold; lưu median, q25–q75, MAD và mẫu thô. Refit đo 15 vòng trên development. Không tính phép so ngưỡng; size joblib là dung lượng file, không phải RAM.')

    page(d,'5. KẾT QUẢ THỰC NGHIỆM')
    best=[t[(t.method==method)&t.feasible].sort_values(['f2','fpr','model_size_mb','latency_ratio'],ascending=[False,True,True,True]).iloc[0] for method in ['random_search','optuna_tpe']]
    labels=['Dummy','RF mặc định, 0,5','RF balanced, 0,5','RF mặc định, OOF τ','RF balanced, OOF τ']
    rows=[[label,*[f'{r[k]:.4f}' for k in ['f2','recall','precision','fpr','threshold']]] for label,(_,r) in zip(labels,b.iterrows())]
    rows += [[f'{r.method} #{int(r.trial)}',*[f'{r[k]:.4f}' for k in ['f2','recall','precision','fpr','threshold']]] for r in best]
    table(d,['Nghiệm OOF','F₂','Recall','Precision','FPR','τ'],rows,9)
    para(d,'Bảng 1. Năm baseline và nghiệm khả thi tốt nhất mỗi phương pháp. Nguồn: baselines.csv, trials.csv; trial từ 0. Số làm tròn 4 chữ số, quyết định dùng độ chính xác đầy đủ trong log.',size=10)
    para(d,f'Chỉ thay ngưỡng RF mặc định nâng F₂ từ {b.iloc[1].f2:.4f} lên {b.iloc[3].f2:.4f}. Tối ưu λ–τ đạt {selected["f2"]:.4f}, tăng {selected["f2"]-b.iloc[3].f2:.4f} so với baseline đã chỉnh ngưỡng. Phần lớn lợi ích so với RF τ=0,5 đến từ ngưỡng; không quy toàn bộ mức tăng cho HPO.')
    heading(d,'5.1. Tính khả thi',2)
    table(d,['Phương pháp','Khả thi','Vi phạm latency','Vi phạm size'],[
        [method,f'{int(t[t.method==method].feasible.sum())}/40',int((~t[t.method==method].latency_ok).sum()),int((~t[t.method==method].size_ok).sum())] for method in ['random_search','optuna_tpe']],10)
    para(d,'Mọi trial đạt FPR 5% sau chọn τ và size 20 MB; các trial bị loại do latency. RS #24 có F₂ 0,7774 nhưng ratio 1,5985× nên không được chọn: nghiệm chất lượng cao nhưng bất khả thi.')
    heading(d,'5.2. Nghiệm cuối và test',2)
    para(d,f'Chọn RS #{selected["trial"]}: {params["n_estimators"]} cây, depth {params["max_depth"]}, split {params["min_samples_split"]}, leaf {params["min_samples_leaf"]}, max_features {params["max_features"]}, class_weight=None, τ={selected["threshold"]:.6f}. Refit đầu tiên hợp lệ, không fallback. Size {test["model_size_mb"]:.3f} MB; ratio {test["latency_ratio"]:.3f}× (q25–q75 {test["latency_ratio_q25"]:.3f}–{test["latency_ratio_q75"]:.3f}×).')
    para(d,f'Test: TN={test["tn"]}, FP={test["fp"]}, FN={test["fn"]}, TP={test["tp"]}; F₂={test["f2"]:.4f}, recall={test["recall"]:.4f}, precision={test["precision"]:.4f}, FPR={test["fpr"]:.4f}, AP={test["average_precision"]:.4f}. Test mô tả nghiệm đã khóa, không chọn cấu hình; lịch sử pilot đã xem test vẫn được công bố.')

    page(d,'5.3. ĐỒ THỊ TỪ LẦN CHẠY THỰC TẾ')
    picture(d,assets/'original_p5_1.jpeg','Hình 1. F₂ OOF khả thi tốt nhất theo trial (trục 1–40). Hình gốc trong ZIP; dữ liệu: trials.csv.')
    para(d,'RS đạt nghiệm cuối ở trial chỉ số 35 (lượt 36); TPE tốt nhất ở chỉ số 2 (lượt 3). Đoạn phẳng là chưa cải thiện nghiệm khả thi tốt nhất, không chứng minh hội tụ toàn cục.',size=11)
    picture(d,assets/'original_p6_1.jpeg','Hình 2. F₂ và FPR; dấu mờ là bất khả thi. Hình gốc trong ZIP; dữ liệu: trials.csv.')
    para(d,'Các điểm ở trái cap FPR 5% nhưng một số vẫn vi phạm latency. F₂–FPR chỉ biểu diễn hai đại lượng; chưa đủ kết luận khả thi nếu bỏ qua size/latency.',size=11)

    page(d,'5.4. HÌNH TRÍCH LOG ĐÃ LƯU')
    picture(d,assets/'original_p7_1.jpeg','Hình 3. Baseline và RS #35; giữ hình chạy thật từ trang 7 PDF trong ZIP.')
    para(d,'Bảng máy đọc tương ứng: results/full/baselines.csv và trials.csv. Số trong ảnh đối chiếu với Bảng 1 và summary.json; CSV/JSON là dữ liệu kiểm chứng chính.',size=11)
    picture(d,assets/'original_p8_1.jpeg','Hình 4. Log test và tài nguyên của nghiệm đã chốt; giữ hình gốc từ trang 8 PDF trong ZIP.')
    para(d,'Ma trận nhầm lẫn có hàng là nhãn thật 0/1 và cột là dự đoán 0/1. final_model.joblib cho phép kiểm tra lại dự đoán; kiểm chứng không huấn luyện lại hoặc sửa τ.',size=11)

    page(d,'5–6. RUNTIME, ỔN ĐỊNH VÀ SENSITIVITY')
    heading(d,'5.5. Chi phí tính toán và độ ổn định',2)
    table(d,['Đại lượng','Giá trị từ log'],[
        ['Wall time cả pipeline',f'{s["wall_seconds"]:.2f} s ({s["wall_seconds"]/60:.2f} phút)'],
        ['Tổng fit 40 trial Random Search',f'{t[t.method=="random_search"].fit_seconds.sum():.2f} s'],
        ['Tổng fit 40 trial TPE',f'{t[t.method=="optuna_tpe"].fit_seconds.sum():.2f} s'],
        ['Refit nghiệm cuối',f'{s["refit_attempts"][-1]["fit_seconds"]:.2f} s'],
        ['Median predict_proba refit',f'{test["latency_ms_per_batch"]:.2f} ms / batch 256']],10.5)
    para(d,'Tổng fit không phải wall time mỗi optimizer: chưa gồm đo latency, chọn ngưỡng, ghi log và overhead. Log không tách wall time từng phương pháp; không suy ra optimizer nhanh/chậm từ tổng fit. Median tỷ số ghép cặp không bằng tỷ số hai median thời gian.',size=11)
    table(d,['Seed mô hình','F₂ OOF','Recall','FPR'],[[int(r.seed),f'{r.f2:.4f}',f'{r.recall:.4f}',f'{r.fpr:.4f}'] for _,r in stab.iterrows()],10.5)
    para(d,f'F₂ trung bình {stab.f2.mean():.4f}, SD mẫu {stab.f2.std(ddof=1):.4f}, khoảng {stab.f2.min():.4f}–{stab.f2.max():.4f}. Giữ λ, τ, fold; chỉ đổi seed RF. Đây không phải ba lần chạy toàn search hoặc ba split độc lập. Chưa có khoảng tin cậy tổng quát cho ưu thế optimizer.',size=11)
    heading(d,'6.1. Độ nhạy theo giới hạn cảnh báo sai',2)
    chosen=sens[(sens.method==selected['method'])&(sens.trial==selected['trial'])]
    table(d,['FPR cap','τ','F₂ OOF','Recall','FP','FN'],[[f'{r.fpr_cap:.0%}',f'{r.threshold:.6f}',f'{r.f2:.4f}',f'{r.recall:.4f}',int(r.fp),int(r.fn)] for _,r in chosen.iterrows()],10.5)
    para(d,'Giữ λ, tối ưu lại τ trên OOF cho từng cap. Cap 2% gần giữ F₂, giảm FP 155→117 nhưng tăng FN 44→51; cap 1% giảm FP còn 74, tăng FN lên 63. Đây là phân tích development; không đổi nghiệm theo test. Ứng dụng thật cần quy định chi phí FP/FN trước khi chọn cap.',size=11)

    page(d,'6. THẢO LUẬN, THẤT BẠI VÀ HẠN CHẾ')
    heading(d,'6.2. Trade-off chất lượng – cảnh báo – tài nguyên',2)
    para(d,'Ngưỡng thấp tăng phát hiện nhưng có thể giảm precision. Nghiệm cuối có recall test 0,8088, song chỉ 55/95 cảnh báo là lỗi thật: 40 báo động sai và 13 lỗi bị bỏ sót. Không thể triển khai tự động chỉ dựa accuracy 0,9735. Có thể hỗ trợ kiểm tra ở điều kiện tương tự dữ liệu, sau khi xác nhận chi phí và tải cảnh báo chấp nhận được.')
    para(d,'RS tốt hơn TPE về F₂ khả thi trong lần chạy này, còn TPE có nhiều trial khả thi hơn. TPE dùng phạt −1, objective không trơn, latency nhiễu; 40 trial có thể chưa đủ học vùng tốt. Đây là diễn giải khả dĩ, chưa kiểm chứng nguyên nhân; không kết luận RS luôn tốt hơn TPE.')
    heading(d,'6.3. Ablation trên development',2)
    table(d,['Đặc trưng','F₂ OOF','FPR','Latency ratio','Size MB'],[['Gốc' if not r.engineered else '+ 3 dẫn xuất',f'{r.f2:.4f}',f'{r.fpr:.4f}',f'{r.latency_ratio:.3f}',f'{r.model_size_mb:.3f}'] for _,r in abl.iterrows()],10)
    para(d,'Ba đặc trưng dẫn xuất: chênh lệch nhiệt độ, công suất cơ học torque × rpm × 2π/60 và torque × tool wear. Chỉ dùng trường cùng thời điểm; giữ λ và τ. OOF tăng lên 0,8462 gợi ý lợi ích tri thức vật lý, nhưng là ablation sau chốt mô hình, không thay nghiệm chính và chưa có test độc lập mới.')
    heading(d,'6.4. Các tình huống chưa được giải quyết',2)
    para(d,'(i) RS #24 F₂ cao hơn nhưng vi phạm latency: loại đúng quy tắc. (ii) Nếu mọi refit không đạt tài nguyên, code báo không có nghiệm khả thi và bỏ qua test, không nới cap. (iii) Chưa biết nguyên nhân vật lý của 13 FN; không gán nguyên nhân bằng suy đoán. (iv) Dữ liệu tổng hợp và split ngẫu nhiên chưa chứng minh hiệu quả trên máy mới, thời gian mới, nhiễu cảm biến hoặc drift.')
    para(d,'OOF vừa chỉnh τ vừa so nhiều trial có thể lạc quan. Pilot cũ đã xem cùng test làm suy yếu tính mù. Ba seed chỉ đo biến thiên mô hình; chưa đo toàn search. Latency phụ thuộc CPU, tải nền và batch; file size không thay RAM. Cần nested evaluation, test độc lập theo thời gian/máy và đo trên phần cứng đích trước triển khai.')

    page(d,'7. KẾT LUẬN VÀ TÁI LẬP')
    heading(d,'7.1. Quyết định cuối cùng',2)
    para(d,f'Chọn RS trial {selected["trial"]}, τ={selected["threshold"]:.6f}, vì F₂ OOF cao nhất trong các quyết định khả thi đã thử và refit đầu tiên cũng đạt tài nguyên. Cấu hình: 100 cây, depth=12, split=10, leaf=2, max_features=0.75, class_weight=None. Chọn theo development trước test; không lấy test quyết định optimizer thắng.')
    para(d,f'F₂ OOF {selected["f2"]:.4f}, FPR OOF {selected["fpr"]:.4f}; size {test["model_size_mb"]:.3f} MB và ratio {test["latency_ratio"]:.3f}×. Test F₂ {test["f2"]:.4f} là kiểm tra hậu chọn với giới hạn lịch sử đã nêu. Đây là nghiệm tốt nhất trong ngân sách/giao thức đã chạy, không phải tối ưu toàn cục hoặc bảo đảm công nghiệp.')
    heading(d,'7.2. Kiểm tra bản tải về',2)
    para(d,'Clone repo hoặc giải nén ZIP tạo bằng git archive từ commit cuối. Cài requirements.txt trong Python 3.14, rồi chạy từ gốc repo:',size=11)
    para(d,'python -m unittest discover -s tests -v\npython scripts/verify_results.py',size=11,bold=True)
    para(d,'Verifier kiểm tra hash CSV, budget, split/fold, thứ tự chọn/refit, mẫu latency, tham số/dự đoán final_model.joblib, số README và nguồn/đầu ra báo cáo. Lệnh kiểm tra không chỉnh file. run.py tạo thư mục mới, không ghi đè results/full. Tái chạy đầy đủ có thể đổi tính khả thi latency do tải hệ thống.',size=11)
    heading(d,'7.3. Minh bạch tiến độ và công cụ',2)
    para(d,'Sáu commit ban đầu cùng ngày 09/11/2026 không chứng minh tiến độ 5 tuần theo đề. Commit hoàn thiện ghi công việc thật ở thời điểm thực hiện; không sửa ngày hoặc dựng lịch sử. Đây là giới hạn không thể khắc phục bằng chỉnh tài liệu. Nộp LMS, pin profile và bảo vệ trực tiếp thuộc phần sinh viên thực hiện.')
    para(d,'OpenAI Codex hỗ trợ rà soát mã, kiểm thử, chuẩn hóa artifact và sinh báo cáo từ log. Sinh viên chịu trách nhiệm đọc hiểu, tái lập và giải thích công thức, ràng buộc, hình và quyết định; báo cáo do công cụ tạo không chứng minh khả năng bảo vệ cá nhân.')
    heading(d,'Thông tin bìa',2)
    para(d,'Họ tên, MSSV và GVHD chép từ trang 1 Bao_Caopdf.pdf trong ZIP. Bản GitHub cũ chỉ để trường chờ điền; không thấy thông tin cụ thể mâu thuẫn. Không có mã lớp/LMS để xác minh. identity.json ghi nguồn; sinh viên cần xác nhận thông tin hành chính trước nộp.',size=11)
    para(d,REPO,size=9)

    page(d,'TÀI LIỆU THAM KHẢO VÀ ĐỐI CHIẾU')
    for ref in [
        '[1] UCI. AI4I 2020 Predictive Maintenance Dataset (2020). DOI: https://doi.org/10.24432/C5HS5C. Dữ liệu tổng hợp; CC BY 4.0: https://creativecommons.org/licenses/by/4.0/.',
        '[2] Bergstra, J. & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. JMLR 13, 281–305. https://jmlr.org/papers/v13/bergstra12a.html',
        '[3] Akiba et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. https://arxiv.org/abs/1907.10902',
        '[4] Tài liệu chính thức: https://scikit-learn.org/stable/modules/ensemble.html#forest và https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html.',
        '[5] Tài liệu môn học được cung cấp: DE_BAI_TAP_LON_CA_NHAN_5_TUAN_GITHUB_PORTFOLIO.pdf (6 trang, đặc biệt trang 4–6); HUONG_DAN_PHAM_VI_DE_BAI_TAP_LON_GITHUB_PORTFOLIO.pdf (5 trang); TAI_LIEU_CHUONG_6_Hyperparameter_Multiobjective_AI.pdf (29 trang). Không tái phân phối các PDF này trong repo.'
    ]: para(d,ref,size=10.5)
    heading(d,'Bằng chứng tương ứng các yêu cầu',2)
    table(d,['Yêu cầu','Vị trí trong bản nộp'],[
        ['7 mục báo cáo','Mục 1–7, trang 3–11'],['Baseline và tối ưu','5 baseline; 40+40 trial; src/experiment.py'],
        ['Bảng, ≥2 đồ thị, log thật','Trang 6–9; results/full CSV/JSON và figures/'],
        ['Runtime, ổn định, sensitivity','Trang 9–10; manifest, stability, sensitivity, ablation'],
        ['Nghiệm và kiểm chứng','Trang 11; final_model.joblib; verify_results.py'],
        ['Word/PDF và bảo vệ','report/bao_cao_do_an_ai4i.docx/.pdf; tóm tắt 1 trang'],
        ['Nguồn và byte/hash','data/README.md; eol_migration.json; build_metadata.json'],
        ['Giới hạn còn lại','Tiến độ 5 tuần; pilot đã xem test; dữ liệu tổng hợp']],9.5)
    para(d,'Logo, motif và 4 hình thực nghiệm giữ từ PDF trong ZIP; provenance ghi ở report/assets/provenance.json. Bảng và số phân tích tính từ log chính thức. Metadata liên kết SHA-256 nguồn với DOCX/PDF cuối; LF áp dụng ở Git và bộ ghi log.',size=10.5)
    d.save(output/'bao_cao_do_an_ai4i.docx')

    brief=new_doc(True); heading(brief,'TÓM TẮT BẢO VỆ • RANDOM FOREST')
    para(brief,f'{identity["student"]} • MSSV {identity["student_id"]} • GVHD {identity["lecturer"]}',size=10)
    heading(brief,'Bài toán và mô hình hóa',2)
    para(brief,'Phát hiện lỗi hiện tại trên AI4I 2020 (UCI, CC BY 4.0; tổng hợp). Tối đa F₂=5TP/(5TP+4FN+FP) theo 6 siêu tham số RF và τ. Ràng buộc OOF: FPR≤5%, size≤20 MB, ratio≤1,25×; kiểm tra lại tài nguyên sau refit.')
    heading(brief,'Thiết kế công bằng',2)
    para(brief,'10.000 hàng; stratified 8.000/2.000, seed 42; 3 fold cố định; one-hot trong fold; bỏ ID/cờ lỗi. 5 baseline, 40 RS+40 TPE (10 startup). Chọn theo OOF/tài nguyên trước test. Pilot cũ từng xem cùng test: không tuyên bố test hoàn toàn mù.')
    table(brief,['Nghiệm OOF','F₂','Recall','FPR'],[
        ['RF mặc định τ=0,5',f'{b.iloc[1].f2:.4f}',f'{b.iloc[1].recall:.4f}',f'{b.iloc[1].fpr:.4f}'],
        ['RF mặc định + OOF τ',f'{b.iloc[3].f2:.4f}',f'{b.iloc[3].recall:.4f}',f'{b.iloc[3].fpr:.4f}'],
        *[[f'{r.method} #{int(r.trial)}',f'{r.f2:.4f}',f'{r.recall:.4f}',f'{r.fpr:.4f}'] for r in best]],9.5)
    heading(brief,'Quyết định và minh chứng',2)
    para(brief,f'RS #35: 100 cây, depth 12, split 10, leaf 2, max_features 0.75, class_weight=None; τ={selected["threshold"]:.6f}. OOF tốt nhất khả thi; refit {test["model_size_mb"]:.3f} MB, {test["latency_ratio"]:.3f}×. Test [[1892,40],[13,55]]: F₂ {test["f2"]:.4f}, recall {test["recall"]:.4f}, precision {test["precision"]:.4f}. Không chọn theo test.')
    para(brief,f'23/40 RS, 31/40 TPE khả thi; runtime {s["wall_seconds"]/60:.2f} phút. Seed 42/43/44: F₂ {stab.f2.mean():.4f} ± {stab.f2.std(ddof=1):.4f} (SD mẫu; giữ fold/τ). Cap 2% giảm FP OOF 155→117, tăng FN 44→51. RS #24 F₂ 0,7774 nhưng ratio 1,5985× nên loại.')
    heading(brief,'Giới hạn và demo 5–7 phút',2)
    para(brief,'Dữ liệu tổng hợp, test đã được pilot xem, một budget/split; chưa chứng minh ưu thế tổng quát. Sáu commit cùng ngày không chứng minh 5 tuần. Demo: formulation (1 phút), pipeline/τ (2), bảng+2 hình+trade-off (2), verifier/giới hạn (1–2).')
    para(brief,'python -m unittest discover -s tests -v\npython scripts/verify_results.py',size=10,bold=True)
    para(brief,REPO,size=8.5)
    para(brief,'Nguồn: results/full; báo cáo 12 trang. Codex hỗ trợ; sinh viên chịu trách nhiệm giải thích.',size=9)
    brief.save(output/'tom_tat_bao_ve_1_trang.docx')
    print('Built DOCX. Export PDFs, inspect layout, then run --finalize-pdfs.')


def finalize(results,output):
    import pymupdf
    names=['bao_cao_do_an_ai4i.docx','bao_cao_do_an_ai4i.pdf','tom_tat_bao_ve_1_trang.docx','tom_tat_bao_ve_1_trang.pdf']
    pages={}
    for name,expected in [(names[1],12),(names[3],1)]:
        with pymupdf.open(output/name) as pdf:
            if len(pdf)!=expected: raise AssertionError(f'{name}: expected {expected} pages, got {len(pdf)}')
            full='\n'.join(p.get_text() for p in pdf)
            if '[CẦN ĐIỀN]' in full: raise AssertionError('Unfilled placeholder')
            for p in pdf:
                for block in p.get_text('blocks'):
                    if block[6]==0 and not p.rect.contains(pymupdf.Rect(block[:4])):
                        raise AssertionError(f'Text outside page {p.number+1}')
            pages[name]=len(pdf)
    sources=[*results.rglob('*'),*output.joinpath('assets').glob('*'),output/'identity.json',
             ROOT/'config.json',ROOT/'requirements.txt',ROOT/'requirements-report.txt',
             ROOT/'scripts/build_report.py',ROOT/'scripts/export_report.ps1',
             *ROOT.joinpath('src').glob('*.py'),ROOT/'data/eol_migration.json',
             ROOT/'ai4i+2020+predictive+maintenance+dataset/ai4i2020.csv']
    metadata={'source_summary_sha256':sha(results/'summary.json'),
              'report_docx':names[0],'report_pdf':names[1],
              'defense_summary_docx':names[2],'defense_summary_pdf':names[3],
              'pages':pages,
              'source_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(sources)) if p.is_file()},
              'output_hashes':{name:sha(output/name) for name in names},
              'note':'PDFs exported from editable DOCX; retained official run, LF bytes, no retraining.'}
    (output/'build_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'pages':pages,'status':'metadata generated from actual bytes'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'results/full')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'report')
    parser.add_argument('--finalize-pdfs',action='store_true')
    args=parser.parse_args()
    (finalize if args.finalize_pdfs else build)(args.results_dir.resolve(),args.output_dir.resolve())
