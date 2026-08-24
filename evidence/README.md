# Bằng chứng — Day 22: LangSmith + Prompt Versioning

Thư mục này chứa bằng chứng chạy thật cho cả 4 nhiệm vụ. Phần dưới đây tập trung phân tích kết quả RAGAS giữa hai phiên bản prompt (yêu cầu ở Nhiệm vụ 3).

## So sánh V1 và V2

| Chỉ số | V1 | V2 | Chênh lệch |
|---|---|---|---|
| faithfulness | 1.0000 | 1.0000 | 0.0000 |
| answer_relevancy | 0.8495 | 0.8501 | +0.0006 (V2) |
| context_recall | 1.0000 | 1.0000 | 0.0000 |
| context_precision | 0.9643 | 0.9677 | +0.0034 (V2) |

Nguồn: `data/ragas_report.json`, chạy trên toàn bộ 50 cặp QA cho mỗi phiên bản, model đánh giá `gemma-4-26b-a4b-it`.

## Hai prompt khác nhau ở đâu

- **V1** (`nguyenhaiquan-rag-v1`): chỉ dẫn ngắn gọn, yêu cầu trả lời 2-4 câu, đi thẳng vào ý chính.
- **V2** (`nguyenhaiquan-rag-v2`): chỉ dẫn theo quy trình ba bước bắt buộc (đọc context, xác định dữ kiện liên quan, viết câu trả lời có tổ chức), yêu cầu 3-5 câu, và nhấn mạnh rõ ràng "mọi khẳng định phải truy được về context, không suy đoán".

## Vì sao V2 nhỉnh hơn

Cả hai phiên bản đều đạt faithfulness và context_recall tuyệt đối, cho thấy knowledge base đủ đầy đủ và retriever (k=3) hoạt động tốt cho cả hai cách diễn đạt prompt — chênh lệch giữa V1 và V2 vì vậy không nằm ở đây.

Chênh lệch thật sự xuất hiện ở answer_relevancy và context_precision, cả hai đều nghiêng nhẹ về V2. Cách giải thích hợp lý nhất là do V2 buộc mô hình liệt kê rõ bước "xác định dữ kiện liên quan tới câu hỏi" trước khi trả lời, thay vì trả lời thẳng như V1. Bước trung gian này hoạt động gần giống một dạng lọc: nó khiến câu trả lời bám sát câu hỏi hơn (answer_relevancy) và giảm khả năng lẫn nội dung không liên quan từ các đoạn context được truy xuất (context_precision). V1 vẫn đạt điểm rất cao ở cả hai chỉ số này, nên chênh lệch chỉ ở mức nhỏ (dưới 0.01), không phải khác biệt lớn.

## Về routing A/B (Nhiệm vụ 2)

50 câu hỏi được định tuyến tất định bằng MD5 hash của `request_id`, cho kết quả V1 = 19 câu, V2 = 31 câu. Vì hash không phụ thuộc thời gian, tỷ lệ này không đổi qua các lần chạy khác nhau — đã kiểm chứng lại nhiều lần trong quá trình làm bài.

## Kết luận

Với knowledge base và bộ câu hỏi hiện tại, V2 là lựa chọn tốt hơn nếu phải chọn một phiên bản duy nhất để triển khai, vì tốt hơn hoặc bằng V1 ở cả 4 chỉ số, không có chỉ số nào bị đánh đổi. Tuy nhiên cách biệt là nhỏ, nên trong thực tế nên cân nhắc thêm chi phí: prompt V2 dài hơn và có thể khiến câu trả lời dài hơn (3-5 câu so với 2-4 câu), tốn nhiều token hơn cho mỗi lượt gọi.
