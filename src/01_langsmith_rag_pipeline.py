"""
Bước 1 — RAG Pipeline với LangSmith Tracing
=============================================
NHIỆM VỤ:
  1. Tải knowledge base, chia chunks, index với FAISS
  2. Xây dựng RAG chain: retriever → prompt → LLM → output parser
  3. Trang trí hàm query với @traceable để LangSmith ghi lại mỗi lần gọi
  4. Chạy 50 câu hỏi → tạo ≥ 50 traces trên LangSmith

DELIVERABLE: Mở https://smith.langchain.com → project của bạn → xác nhận ≥ 50 traces.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ⚠️ QUAN TRỌNG: Import config TRƯỚC KHI import bất kỳ thư viện LangChain nào.
# config.py tự động đặt LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, ... vào os.environ
import config

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text
from qa_pairs import SAMPLE_QUESTIONS

# Cache FAISS index cục bộ (đường dẫn tương đối, không commit — xem .gitignore).
# NV1/NV2/NV3 đều embed đúng 107 chunks giống hệt nhau; không cache thì mỗi lần
# chạy lại tốn thêm ~107 lượt gọi embedding, dễ chạm giới hạn free tier theo ngày.
_FAISS_CACHE_DIR = Path(__file__).parent.parent / "data" / ".faiss_cache"


def _build_vectorstore_with_retry(chunks, embeddings, batch_size: int = 50, max_retries: int = 5):
    """
    Dựng FAISS vectorstore theo từng lô nhỏ (mặc định 50 chunks/lô) thay vì gộp
    hết vào một lần gọi embed_documents().

    Lý do: free tier của Gemini giới hạn ~100 đoạn văn bản được embed mỗi phút.
    Gộp toàn bộ chunks vào 1 lần gọi thì bị vượt giới hạn NGAY TRONG lần gọi đó
    (không phải do gọi dồn dập nhiều lần) — nên retry/chờ không giải quyết được,
    phải giảm số lượng văn bản mỗi lần gọi xuống dưới ngưỡng.

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


# ── 1. Thiết lập Vectorstore ───────────────────────────────────────────────
def setup_vectorstore():
    """
    Tải knowledge base, chia chunks và tạo FAISS vectorstore.

    Gợi ý:
        embeddings  = get_embeddings()
        text        = load_knowledge_base()
        chunks      = split_text(text, chunk_size=500, chunk_overlap=50)
        vectorstore = build_vectorstore(chunks, embeddings)
    """
    embeddings = get_embeddings()

    text = load_knowledge_base()

    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    print(f"📚 Đã chia thành {len(chunks)} chunks")

    vectorstore = _build_vectorstore_with_retry(chunks, embeddings)
    return vectorstore


# ── 2. RAG Prompt Template ─────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Bạn là trợ lý AI hữu ích. Chỉ dùng context sau để trả lời.\n\nContext:\n{context}"),
    ("human",  "{question}"),
])


# ── 3. Build RAG Chain ─────────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Xây dựng LCEL RAG chain theo cấu trúc pipe:
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()

    Trả về: (chain, retriever)
    """
    llm = get_llm()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        """Ghép nội dung các Document thành một khối text cho biến {context}."""
        return "\n\n".join(doc.page_content for doc in docs)

    # Dict ở đầu chain chạy song song hai nhánh trên cùng một input:
    # nhánh context đi qua retriever rồi format_docs, nhánh question giữ nguyên câu hỏi
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ── 4. Hàm Query có LangSmith Tracing ─────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str, max_retries: int = 5) -> str:
    """
    Chạy RAG chain với một câu hỏi.
    Decorator @traceable sẽ gửi mỗi lần gọi lên LangSmith như một trace riêng.

    Có retry cho lỗi 429 (rate-limit) vì free tier của provider miễn phí
    thường giới hạn theo request/phút — 50 câu hỏi liên tiếp có thể chạm giới hạn.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke(question)
        except Exception as e:
            msg = str(e)
            is_rate_limit = "RESOURCE_EXHAUSTED" in msg or "429" in msg
            if not is_rate_limit or attempt == max_retries:
                raise
            print(f"    ⏳ Dính rate-limit (lần {attempt}/{max_retries}), chờ 65s...")
            time.sleep(65)


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 1: LangSmith RAG Pipeline")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()

    chain, retriever = build_rag_chain(vectorstore)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
        print(f"       A: {str(answer)[:100]}\n")

    print(f"\n✅ {len(SAMPLE_QUESTIONS)} traces đã gửi lên LangSmith project '{config.LANGSMITH_PROJECT}'")
    print("   Mở https://smith.langchain.com để xem traces.")


if __name__ == "__main__":
    main()
