"""
src/llm.py  –  Gọi LLM (có cache) để sinh thought cho ToT ở GĐ6.

Mọi thiết lập nằm trong file .env ở gốc repo (không đưa lên GitHub):
    LLM_BACKEND   = mock | openai_compat
    LLM_BASE_URL  = vd http://localhost:11434/v1 (Ollama) hoặc https://api.deepseek.com
    LLM_API_KEY   = key của nhà cung cấp (Ollama thì điền gì cũng được)
    LLM_MODEL     = tên mô hình, vd llama31-8k hoặc deepseek-flash

Hai backend:
- "mock": LLM giả, trả về 3 cấu hình ngẫu nhiên trong thực đơn SEARCH_SPACE_V2. Chỉ để kiểm tra pipeline,
  kết quả KHÔNG có ý nghĩa nghiên cứu (bản chất là random search).
- "openai_compat": gọi endpoint /chat/completions theo chuẩn OpenAI. Ollama và DeepSeek
  đều hỗ trợ chuẩn này -> đổi LLM chỉ cần sửa .env.

Cache: mỗi bộ (backend, model, system, prompt, temperature, max_tokens, tag) chỉ gọi LLM
MỘT lần; câu trả lời lưu vào results/llm_cache.json. Nhờ vậy:
  (1) chạy lại notebook cho đúng kết quả cũ -> tái lập được,
  (2) không tốn tiền/thời gian gọi lại.
"""
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request

from src.data import ROOT

CACHE_FILE = ROOT / "results" / "llm_cache.json"
STATS = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
_cache = None


# ----------------------------------------------------------------------------
# 1. Đọc thiết lập từ .env
# ----------------------------------------------------------------------------
def settings():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=True)
    except ImportError:
        pass  # không có python-dotenv thì chỉ đọc biến môi trường sẵn có
    return {"backend": os.getenv("LLM_BACKEND", "mock"),
            "base_url": os.getenv("LLM_BASE_URL", "").rstrip("/"),
            "api_key": os.getenv("LLM_API_KEY", ""),
            "model": os.getenv("LLM_MODEL", "mock")}


# ----------------------------------------------------------------------------
# 2. Cache
# ----------------------------------------------------------------------------
def _get_cache():
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}
    return _cache


def _save_cache():
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(_cache, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(CACHE_FILE)  # ghi file tạm rồi đổi tên -> không hỏng cache nếu bị ngắt giữa chừng


def _cache_key(s, system, prompt, temperature, max_tokens, tag):
    raw = json.dumps([s["backend"], s["model"], system, prompt, temperature, max_tokens, tag],
                     ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# 3. Hai backend
# ----------------------------------------------------------------------------
def _mock_reply(key):
    """LLM giả: 3 cấu hình ngẫu nhiên trong thực đơn (seed theo key -> lần nào cũng như nhau)."""
    from src.models import all_configs, SEARCH_SPACE_V2
    rng = random.Random(key)
    configs = rng.sample(all_configs(SEARCH_SPACE_V2), 3)
    thoughts = [{"hypothesis": {"category": "mock", "statement": "cấu hình ngẫu nhiên", "evidence": []},
                 "config": c} for c in configs]
    return json.dumps({"thoughts": thoughts}, ensure_ascii=False), {"model": "mock"}


def _openai_compat_reply(s, system, prompt, temperature, max_tokens, json_mode,
                         timeout=600, retries=3):
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    body = {"model": s["model"], "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}  # yêu cầu trả về JSON hợp lệ
    req = urllib.request.Request(
        s["base_url"] + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {s['api_key']}"})

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read().decode("utf-8"))
            text = resp["choices"][0]["message"]["content"]
            meta = {"model": resp.get("model", s["model"]),  # phiên bản THẬT mà server trả về
                    "usage": resp.get("usage", {})}
            return text, meta
        except urllib.error.HTTPError as e:
            # 429 (quá nhiều request) và lỗi 5xx của server thì thử lại; lỗi khác (sai key, 402 hết tiền) thì dừng
            if e.code in (429, 500, 502, 503) and attempt < retries:
                time.sleep(5 * attempt)
                continue
            raise RuntimeError(f"LLM trả lỗi HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:300]}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries:
                time.sleep(5 * attempt)
                continue
            raise RuntimeError(f"Không kết nối được LLM tại {s['base_url']}: {e}")


# ----------------------------------------------------------------------------
# 4. Hàm chính
# ----------------------------------------------------------------------------
def call_llm(prompt, system="", temperature=0.7, max_tokens=600, tag="", json_mode=True,
             use_cache=True):
    """Gửi prompt, trả về chuỗi trả lời của LLM.

    tag: đổi tag để lấy một mẫu KHÁC cho cùng prompt (vd tag="rep2" cho lần lặp thứ 2),
         vì cache sẽ trả lại đúng câu trả lời cũ nếu mọi thứ giống hệt.
    """
    s = settings()
    key = _cache_key(s, system, prompt, temperature, max_tokens, tag)
    cache = _get_cache()
    if use_cache and key in cache:
        STATS["cache_hits"] += 1
        return cache[key]["text"]

    t0 = time.time()
    if s["backend"] == "mock":
        text, meta = _mock_reply(key)
    elif s["backend"] == "openai_compat":
        text, meta = _openai_compat_reply(s, system, prompt, temperature, max_tokens, json_mode)
    else:
        raise ValueError(f"LLM_BACKEND không hợp lệ: {s['backend']}")
    dt = time.time() - t0

    STATS["calls"] += 1
    STATS["seconds"] += dt
    usage = meta.get("usage", {})
    STATS["prompt_tokens"] += usage.get("prompt_tokens", 0)
    STATS["completion_tokens"] += usage.get("completion_tokens", 0)

    cache[key] = {"text": text, "model_returned": meta.get("model"), "usage": usage,
                  "seconds": round(dt, 2), "backend": s["backend"], "model": s["model"],
                  "temperature": temperature, "tag": tag,
                  "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save_cache()
    return text


# ----------------------------------------------------------------------------
# 5. Đọc JSON từ câu trả lời
# ----------------------------------------------------------------------------
def extract_json(text):
    """Lấy object JSON từ câu trả lời của LLM.

    Mô hình nhỏ hay bọc JSON trong ```json ... ``` hoặc nói thêm vài câu trước/sau,
    nên: (1) gỡ khối ```, (2) cắt từ dấu { đầu tiên đến dấu } cuối cùng, (3) json.loads.
    Không đọc được thì báo ValueError để nơi gọi quyết định (bỏ thought hay gọi lại).
    """
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"Không tìm thấy JSON trong câu trả lời: {text[:200]!r}")
    try:
        return json.loads(t[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON lỗi ({e}): {text[:200]!r}")
