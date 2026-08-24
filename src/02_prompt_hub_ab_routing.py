"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
import sys
import time
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text
from qa_pairs import SAMPLE_QUESTIONS

# Cache FAISS index cục bộ (đường dẫn tương đối, không commit — xem .gitignore).
# NV1/NV2/NV3 đều embed đúng 107 chunks giống hệt nhau; không cache thì mỗi lần
# chạy lại tốn thêm ~107 lượt gọi embedding, dễ chạm giới hạn free tier theo ngày.
_FAISS_CACHE_DIR = Path(__file__).parent.parent / "data" / ".faiss_cache"


def _build_vectorstore_with_retry(chunks, embeddings, batch_size: int = 50, max_retries: int = 5):
    """
    Dựng FAISS vectorstore theo từng lô nhỏ để tránh vượt giới hạn free tier
    (~100 đoạn văn bản embed/phút của Gemini). Xem giải thích chi tiết ở
    01_langsmith_rag_pipeline.py.

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


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
# Tên phải duy nhất trong Hub của tài khoản LangSmith đang dùng
PROMPT_V1_NAME = "nguyenhaiquan-rag-v1"
PROMPT_V2_NAME = "nguyenhaiquan-rag-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
# V1 — phong cách ngắn gọn, ưu tiên trả lời thẳng và ít chữ
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

# V2 — phong cách chuyên gia, buộc mô hình bám sát từng dữ kiện trong context
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


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def push_prompts_to_hub(client: Client):
    """
    Upload cả 2 prompt templates lên LangSmith Prompt Hub.
    Gợi ý: client.push_prompt(name, object=template, description="...")
    """
    try:
        url = client.push_prompt(
            PROMPT_V1_NAME,
            object=PROMPT_V1,
            description="V1 - phong cách ngắn gọn, 2-4 câu",
        )
        print(f"✅ Đã push V1 → {url}")
    except Exception as e:
        print(f"⚠️  V1 lỗi: {e}")

    try:
        url = client.push_prompt(
            PROMPT_V2_NAME,
            object=PROMPT_V2,
            description="V2 - phong cách chuyên gia, có cấu trúc, 3-5 câu",
        )
        print(f"✅ Đã push V2 → {url}")
    except Exception as e:
        print(f"⚠️  V2 lỗi: {e}")


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """
    Tải 2 prompt từ LangSmith Prompt Hub.
    Fallback về template local nếu Hub không khả dụng.

    Gợi ý: client.pull_prompt(name) → ChatPromptTemplate

    Trả về: {name: ChatPromptTemplate}
    """
    prompts = {}

    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Đã pull '{PROMPT_V1_NAME}' từ Hub")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V1_NAME}'")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Đã pull '{PROMPT_V2_NAME}' từ Hub")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V2_NAME}'")

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Xác định prompt version dựa trên MD5 hash của request_id.

    Quy tắc: hash chẵn → PROMPT_V1_NAME | hash lẻ → PROMPT_V2_NAME
    TÍNH CHẤT: cùng request_id LUÔN cho cùng kết quả (deterministic).

    Gợi ý:
        hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME
    """
    # MD5 là hàm băm tất định: cùng request_id luôn ra cùng chuỗi hex,
    # nên việc phân nhánh không phụ thuộc vào thời điểm hay lần chạy
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)

    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Chạy RAG chain với prompt version được chọn bởi router.

    Bước:
      a) Retrieve top-3 docs từ retriever
      b) Ghép page_content thành context string
      c) Chạy (prompt | llm | StrOutputParser()).invoke({"context": ..., "question": ...})
      d) Trả về {"question": ..., "answer": ..., "version": ...}
    """
    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    # Retrieve nằm trong hàm được @traceable bọc, nên trace trên LangSmith
    # chứa cả câu hỏi, context truy xuất được lẫn câu trả lời.
    # Có retry cho lỗi 429 (rate-limit) vì free tier có thể chạm giới hạn
    # khi chạy liên tiếp 50 câu hỏi.
    chain = prompt | llm | StrOutputParser()
    answer = None
    for attempt in range(1, 6):
        try:
            answer = chain.invoke({"context": context, "question": question})
            break
        except Exception as e:
            msg = str(e)
            is_rate_limit = "RESOURCE_EXHAUSTED" in msg or "429" in msg
            if not is_rate_limit or attempt == 5:
                raise
            print(f"    ⏳ Dính rate-limit (lần {attempt}/5), chờ 65s...")
            time.sleep(65)

    return {"question": question, "answer": answer, "version": version}


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return _build_vectorstore_with_retry(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    client = Client(api_key=config.LANGSMITH_API_KEY)

    push_prompts_to_hub(client)

    # Prompt dùng khi chạy được lấy về từ Hub, không dùng thẳng biến local
    prompts = pull_prompts_from_hub(client)

    # Tạo vectorstore, retriever và LLM
    vectorstore = setup_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    # Chạy A/B routing cho tất cả câu hỏi
    v1_count, v2_count = 0, 0
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"

        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        result = ask_ab(retriever, llm, prompt, question, version_tag)

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}...")

    print(f"\n📊 Routing: V1={v1_count} câu | V2={v2_count} câu | Tổng={len(SAMPLE_QUESTIONS)}")
    print("✅ Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")


if __name__ == "__main__":
    main()
