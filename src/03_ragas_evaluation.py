"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text
from qa_pairs import QA_PAIRS

# Cache FAISS index cục bộ (đường dẫn tương đối, không commit — xem .gitignore).
# NV1/NV2/NV3 đều embed đúng 107 chunks giống hệt nhau; không cache thì mỗi lần
# chạy lại tốn thêm ~107 lượt gọi embedding, dễ chạm giới hạn free tier theo ngày.
_FAISS_CACHE_DIR = Path(__file__).parent.parent / "data" / ".faiss_cache"


def _build_vectorstore_with_retry(chunks, embeddings, batch_size: int = 50, max_retries: int = 5):
    """
    Dựng FAISS vectorstore theo từng lô nhỏ để tránh vượt giới hạn free tier
    (~100 đoạn văn bản embed/phút của Gemini). Xem giải thích chi tiết ở
    01_langsmith_rag_pipeline.py — cùng một lỗi 429 gặp phải khi gộp hết
    chunks vào một lần gọi embed_documents().

    Có cache trên đĩa: nếu đã dựng trước đó (bởi NV1/NV2/NV3 bất kỳ), đọc lại từ
    data/.faiss_cache/ thay vì gọi embedding lại, để không tốn thêm quota.
    """
    from langchain_community.vectorstores import FAISS

    if _FAISS_CACHE_DIR.exists():
        try:
            vectorstore = FAISS.load_local(
                str(_FAISS_CACHE_DIR), embeddings, allow_dangerous_deserialization=True
            )
            print(f"  💾 Đã tải FAISS index từ cache ({_FAISS_CACHE_DIR}) — không tốn quota embedding")
            return vectorstore
        except Exception as e:
            print(f"  ⚠️  Cache lỗi ({e}), dựng lại từ đầu")

    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    vectorstore = None
    for bi, batch in enumerate(batches, 1):
        for attempt in range(1, max_retries + 1):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_texts(batch, embeddings)
                else:
                    vectorstore.add_texts(batch)
                print(f"  ✅ Lô {bi}/{len(batches)} đã embed xong ({len(batch)} chunks)")
                break
            except Exception as e:
                msg = str(e)
                is_rate_limit = "RESOURCE_EXHAUSTED" in msg or "429" in msg
                if not is_rate_limit or attempt == max_retries:
                    raise
                print(f"  ⏳ Dính rate-limit ở lô {bi} (lần {attempt}/{max_retries}), chờ 65s...")
                time.sleep(65)

    try:
        _FAISS_CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(_FAISS_CACHE_DIR))
        print(f"  💾 Đã lưu FAISS index vào cache để lần chạy sau không tốn quota embedding")
    except Exception as e:
        print(f"  ⚠️  Không lưu được cache ({e}), bỏ qua")

    return vectorstore


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
# Giữ nguyên văn hai system prompt đã push lên Hub ở Bước 2, để điểm RAGAS
# đo đúng hai phiên bản đang chạy A/B chứ không phải một biến thể khác
SYSTEM_V1 = (
    "Bạn là trợ lý AI hữu ích. Chỉ dùng context dưới đây để trả lời, "
    "tuyệt đối không thêm kiến thức bên ngoài.\n"
    "Giữ câu trả lời ngắn gọn, tối đa 2-4 câu, đi thẳng vào ý chính.\n"
    "Nếu context không chứa thông tin cần thiết, hãy nói rõ là không tìm thấy trong tài liệu.\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích tài liệu. Quy trình bắt buộc:\n"
    "1. Đọc kỹ toàn bộ context bên dưới.\n"
    "2. Xác định những dữ kiện liên quan trực tiếp tới câu hỏi.\n"
    "3. Viết câu trả lời rõ ràng, có tổ chức, dài 3-5 câu.\n"
    "Mọi khẳng định phải truy được về context; không suy đoán, không bổ sung "
    "kiến thức ngoài tài liệu.\n"
    "Nếu context thiếu thông tin, hãy nêu rõ phần nào còn thiếu thay vì đoán.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return _build_vectorstore_with_retry(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)

    # Giữ từng đoạn tách rời: context_precision và context_recall chấm điểm
    # trên từng đoạn một, ghép chuỗi ở đây sẽ biến 3 đoạn thành 1 và làm sai số đo
    contexts = [doc.page_content for doc in docs]

    # Chỉ ghép khi đưa vào biến {context} của prompt
    ctx_str = "\n\n".join(contexts)

    chain = prompt | llm | StrOutputParser()
    answer = None
    for attempt in range(1, 6):
        try:
            answer = chain.invoke({"context": ctx_str, "question": question})
            break
        except Exception as e:
            msg = str(e)
            is_rate_limit = "RESOURCE_EXHAUSTED" in msg or "429" in msg
            if not is_rate_limit or attempt == 5:
                raise
            print(f"    ⏳ Dính rate-limit (lần {attempt}/5), chờ 65s...")
            time.sleep(65)

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    dataset = build_ragas_dataset(rag_results)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    # answer_relevancy mặc định strictness=3: yêu cầu model sinh 3 phương án
    # trong 1 lần gọi (n=3). Các model Google (Gemini lẫn Gemma) không hỗ trợ
    # n>1 qua langchain_google_genai — trả lỗi "Multiple candidates is not
    # enabled for this model" ở MỌI job liên quan tới chỉ số này. Giảm xuống 1
    # để mỗi lần chỉ xin 1 phương án, né lỗi hoàn toàn (đánh đổi: ước lượng độ
    # liên quan dựa trên 1 câu hỏi sinh ngược thay vì trung bình 3 câu).
    answer_relevancy.strictness = 1

    # RAGAS 0.4: llm và embeddings truyền vào evaluate(), không truyền vào metric.
    # max_workers mặc định là 16 lượt gọi song song — quá cao cho free tier của
    # provider miễn phí, dễ dồn dập vượt giới hạn request/phút. Giảm xuống 3 để
    # các lượt gọi rải ra, max_retries/max_wait giữ mặc định để tự chờ khi cần.
    run_config = RunConfig(max_workers=3)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
        run_config=run_config,
    )

    # Tính mean score cho mỗi metric, bỏ qua cả None lẫn NaN.
    # Lưu ý quan trọng: faithfulness trả về NaN (không phải exception) cho mẫu
    # nào mà bước trích statement không lấy ra được câu nào — np.mean thường sẽ
    # lan NaN ra toàn bộ kết quả chỉ vì 1 mẫu lỗi, nên phải lọc bằng np.isnan()
    # trước khi tính trung bình để không mất hết điểm từ các mẫu hợp lệ còn lại.
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        valid = [v for v in raw if v is not None and not (isinstance(v, float) and np.isnan(v))]
        n_dropped = len(raw) - len(valid)
        if n_dropped:
            print(f"  ⚠️  {key}: bỏ qua {n_dropped}/{len(raw)} mẫu không tính được điểm (NaN/None)")
        scores[key] = float(np.mean(valid)) if valid else float("nan")

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()

    # Thu thập kết quả RAG cho cả V1 và V2
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()
