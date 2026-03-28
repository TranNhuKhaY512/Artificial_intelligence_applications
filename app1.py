
import os
import re
import json
import unicodedata
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from flask import Flask, render_template, request, jsonify, send_from_directory
import google.generativeai as genai

# ======================
# GOOGLE GEMINI CONFIG
# ======================
GEMINI_API_KEY = "YOUR_API_KEY" 
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash") 

def call_gemini(prompt: str) -> str:
    try:
        res = gemini_model.generate_content(prompt)
        return res.text or "Không có phản hồi từ Gemini."
    except Exception as e:
        return f"❌ Gemini API error: {str(e)}"

# ======================
# 1) Đọc & xử lý dữ liệu
# ======================
CSV_PATH = "REAL_ESTATE.csv"
IMG_FOLDER = "hinhanh/hinhanh"
df = pd.read_csv(CSV_PATH)
  
# 
def normalize_price(v):
    if pd.isna(v): return np.nan
    s = str(v).lower().strip().replace(",", "").replace(" ", "")
    s = s.replace("ty", "tỷ").replace("trieu", "triệu")
    if "tỷ" in s:
        return float(re.sub(r"[^0-9.]", "", s) or 0) * 1_000_000_000
    if "triệu" in s:
        return float(re.sub(r"[^0-9.]", "", s) or 0) * 1_000_000
    s = re.sub(r"[^0-9.]", "", s)
    return float(s) if s else np.nan

df["Giá"] = df["Giá"].apply(normalize_price)
df["Địa chỉ"] = df["Địa chỉ"].astype(str).str.strip()

def find_image_path(_id):
    _id = str(_id)
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        p = os.path.join(IMG_FOLDER, _id + ext)
        if os.path.exists(p):
            return f"/hinhanh/{_id}{ext}" 
    return None

df["image_path"] = df["Id"].apply(find_image_path)

# ======================
# 2) Nhận diện & Chuẩn hóa Quận chính xác
# ======================
DISTRICT_MAP = {
    "q1": "Quận 1", "q 1": "Quận 1", "quận1": "Quận 1", "quan 1": "Quận 1",
    "q2": "Quận 2", "q 2": "Quận 2", "quận2": "Quận 2", "quan 2": "Quận 2",
    "q3": "Quận 3", "q 3": "Quận 3", "quận3": "Quận 3", "quan 3": "Quận 3",
    "q4": "Quận 4", "q 4": "Quận 4", "quận4": "Quận 4", "quan 4": "Quận 4",
    "q5": "Quận 5", "q 5": "Quận 5", "quận5": "Quận 5", "quan 5": "Quận 5",
    "q6": "Quận 6", "q 6": "Quận 6", "quận6": "Quận 6", "quan 6": "Quận 6",
    "q7": "Quận 7", "q 7": "Quận 7", "quận7": "Quận 7", "quan 7": "Quận 7",
    "q8": "Quận 8", "q 8": "Quận 8", "quận8": "Quận 8", "quan 8": "Quận 8",
    "q9": "Quận 9", "q 9": "Quận 9", "quận9": "Quận 9", "quan 9": "Quận 9",
    "q10": "Quận 10", "q 10": "Quận 10", "quận10": "Quận 10", "quan 10": "Quận 10",
    "q11": "Quận 11", "q 11": "Quận 11", "quận11": "Quận 11", "quan 11": "Quận 11",
    "q12": "Quận 12", "q 12": "Quận 12", "quận12": "Quận 12", "quan 12": "Quận 12",
    "binh thanh": "Bình Thạnh",
    "go vap": "Gò Vấp",
    "phu nhuan": "Phú Nhuận",
    "thu duc": "Thủ Đức",
    "tan binh": "Tân Bình",
    "tan phu": "Tân Phú",
    "binh tan": "Bình Tân",
    "nha be": "Nhà Bè",
    "hoc mon": "Hóc Môn",
    "binh chanh": "Bình Chánh",
    "cu chi": "Củ Chi"
}

def remove_accents(input_str: str) -> str:
    """Xóa dấu tiếng Việt, chuyển về chữ thường"""
    if not isinstance(input_str, str):
        return ""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def extract_district(addr: str) -> str:
    """Chuẩn hóa tên quận từ địa chỉ (bắt cả Q7, Q.7, quan7, Bình Thạnh, v.v.)."""
    if not isinstance(addr, str) or not addr.strip():
        return "Khác"
        
    s = remove_accents(addr.lower())
    s_no_space = s.replace(" ", "")
    s = re.sub(r"[.,\-_/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    m = re.search(r"\bq[\s\.]*(\d{1,2})\b", s)
    if m:
        return f"Quận {m.group(1)}"
    m2 = re.search(r"quan\s*(\d{1,2})", s)
    if m2:
        return f"Quận {m2.group(1)}"

    for key, val in DISTRICT_MAP.items():
        key_no_space = key.replace(" ", "")
        if key_no_space in s_no_space or key in s:
             return val

    if "binhthanh" in s_no_space or "bthanh" in s_no_space:
        return "Bình Thạnh"
    if "thuduc" in s_no_space:
        return "Thủ Đức"
    if "q7" in s_no_space or "quan7" in s_no_space:
        return "Quận 7"

    return "Khác"


df["district"] = df["Địa chỉ"].apply(extract_district)
df["district"] = df["district"].str.title()
df["district_lower"] = df["district"].str.lower()
df["district_no_accent"] = df["district"].apply(lambda x: remove_accents(x).lower().replace(" ", ""))

df = df.dropna(subset=["Giá", "Số phòng ngủ"]).reset_index(drop=True)

top_districts = df['district'].value_counts().nlargest(30).index.tolist()
df['district_reduced'] = df['district'].where(df['district'].isin(top_districts), other='OTHER')

# ========== CHUẨN HÓA TÍNH NĂNG ==========
num_features = df[["Giá", "Số phòng ngủ"]].astype(float)
scaler = StandardScaler().fit(num_features)
num_scaled = scaler.transform(num_features)

try:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except:
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")

cat_encoded = ohe.fit_transform(df[["district_reduced"]])
X = np.hstack([num_scaled, cat_encoded])
nbrs = NearestNeighbors(n_neighbors=6, metric="euclidean").fit(X)

districts_all = sorted(df["district"].unique().tolist())

# ======================
# 3) Tiện ích xử lý
# ======================
def format_gia(gia: float) -> str:
    if gia >= 1_000_000_000:
        return f"{gia/1_000_000_000:.2f} tỷ"
    if gia >= 1_000_000:
        return f"{gia/1_000_000:.0f} triệu"
    return f"{gia:.0f} đ"

# ======================
# 4) Parse truy vấn 
# ======================
def parse_query(q: str) -> dict:
    q_lower = q.lower().strip()

    q_no_accent_no_space = remove_accents(q_lower).replace(" ", "")
    q_no_accent = remove_accents(q_lower)
    q_no_accent = re.sub(r"[.,\-_/]", " ", q_no_accent)
    q_no_accent = re.sub(r"\s+", " ", q_no_accent).strip()

    bedrooms, price_min, price_max = None, None, None
    districts, property_type = [], []
        
    # ⭐️ BẮT SỐ LƯỢNG CĂN HỘ MUỐN GỢI Ý
    num_to_recommend = 1
    m_num = re.search(r"(?:goi y|chon)\s*(?:cho toi)?\s*(\d+|mot|hai|ba|bon|nam|sau)\s*căn", q_no_accent)
    
    if m_num:
        num_str = m_num.group(1).lower()
        num_map = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "nam": 5, "sau": 6}
        num_to_recommend = num_map.get(num_str) or (int(num_str) if num_str.isdigit() else 1)
        num_to_recommend = min(num_to_recommend, 6) 


    # Bắt phòng ngủ
    bedrooms_found = re.findall(r"(\d+)\s*phong", q_no_accent)
    bedrooms = [int(b) for b in bedrooms_found if b.isdigit()]
    bedrooms = bedrooms if bedrooms else None 

    # Bắt giá
    
    # Hàm chuyển đổi giá trị và đơn vị
    def convert_price_to_vnd(value, unit):
        try:
            val = float(value)
            if unit in ["ty", "tỷ"]:
                return val * 1_000_000_000
            elif unit in ["trieu", "triệu"]:
                return val * 1_000_000
            return val
        except:
            return 0
            
    price_patterns = [
        # 1. Từ A đến B (ví dụ: từ 10 đến 15 tỷ)
        (r"tu\s*([\d\.]+)\s*(ty|trieu)?\s*den\s*([\d\.]+)\s*(ty|trieu)?", "range_ab"),
        # 2. Giá tầm X (ví dụ: tầm 20 tỷ)
        (r"gia\s*tam\s*([\d\.]+)\s*(ty|trieu)?", "around"),
        # 3. Dưới X, Tối đa X (ví dụ: dưới 5 tỷ)
        (r"(?:duoi|toi\s*da|max)\s*([\d\.]+)\s*(ty|trieu)?", "max"),
        # 4. Trên Y, Tối thiểu Y (ví dụ: trên 10 tỷ)
        (r"(?:tren|toi\s*thieu|min)\s*([\d\.]+)\s*(ty|trieu)?", "min"),
        # 5. Giá X (Mặc định là max) (ví dụ: 5 tỷ)
        (r"([\d\.]+)\s*(ty|trieu)", "max_default")
    ]
    
    for pattern, mode in price_patterns:
        m = re.search(pattern, q_no_accent)
        if m:
            if mode == "range_ab":
                val1 = convert_price_to_vnd(m.group(1), m.group(2))
                val2 = convert_price_to_vnd(m.group(3), m.group(4))
                price_min, price_max = min(val1, val2), max(val1, val2)
            elif mode == "around":
                center_price = convert_price_to_vnd(m.group(1), m.group(2))
                # Biên độ 15%
                margin = center_price * 0.15
                price_min = center_price - margin
                price_max = center_price + margin
            elif mode == "max": 
                price_max = convert_price_to_vnd(m.group(1), m.group(2))
            elif mode == "min": 
                price_min = convert_price_to_vnd(m.group(1), m.group(2))
            elif mode == "max_default": 
                price_max = convert_price_to_vnd(m.group(1), m.group(2))
            break # Lấy kết quả bắt được đầu tiên

    # Bắt nhiều quận cùng lúc
    found = set()
    for num in re.findall(r"quan\s*(\d{1,2})", q_no_accent):
        found.add(f"Quận {int(num)}")
    for key, val in DISTRICT_MAP.items():
        key_no_space = key.replace(" ", "")
        if key_no_space in q_no_accent_no_space:
            found.add(val)
    districts = list(found)

    # Bắt loại hình
    property_keywords = ["biet thu", "can ho", "chung cu", "dat", "nha pho", "shophouse"]
    property_type_raw = [kw for kw in property_keywords if kw in q_no_accent]
    property_map_back = {"biet thu": "biệt thự", "can ho": "căn hộ", "chung cu": "chung cư", 
                         "dat": "đất", "nha pho": "nhà phố", "shophouse": "shophouse"}
    property_type = [property_map_back[kw] for kw in property_type_raw]
    
    if not property_type:
        property_keywords_full = ["biệt thự", "căn hộ", "chung cư", "đất", "nhà phố", "shophouse"]
        property_type = [kw for kw in property_keywords_full if kw in q_lower]

    return {
        "bedrooms": bedrooms, 
        "districts": districts,
        "price_min": price_min,
        "price_max": price_max,
        "property_type": property_type,
        "num_to_recommend": num_to_recommend 
    }


# ======================
# 5) Gợi ý bằng KNN 
# ======================
def recommend_by_features(Giá=None, districts=None, bedrooms=None, property_type=None, price_min=None, top_k=6):
    df_f = df.copy()
    
    # ⭐️ LOGIC LỌC GIÁ MỚI: Sử dụng mask rõ ràng cho min và max
    price_mask = pd.Series(True, index=df_f.index)
    
    # 1. Áp dụng giới hạn dưới (price_min)
    if price_min is not None:
        price_mask &= (df_f["Giá"] >= price_min)
        
    # 2. Áp dụng giới hạn trên (Giá tương đương price_max)
    if Giá is not None:
        price_mask &= (df_f["Giá"] <= Giá)
        
    df_f = df_f[price_mask].copy() 
    
    if df_f.empty: return []

    df_temp = df_f.copy() 
    if bedrooms and isinstance(bedrooms, list) and len(bedrooms) > 0:
        df_temp = df_temp[df_temp["Số phòng ngủ"].isin(bedrooms)]
    elif bedrooms and (isinstance(bedrooms, int) or isinstance(bedrooms, float)):
        df_temp = df_temp[df_temp["Số phòng ngủ"] == bedrooms]

    if districts:
        lower_districts = [d.lower().strip() for d in districts]
        df_temp = df_temp[df_temp["district_lower"].apply(lambda x: any(d in x for d in lower_districts))]

    if property_type:
        pattern = "|".join(map(re.escape, property_type))
        df_temp = df_temp[df_temp["Tên"].str.contains(pattern, case=False, na=False) |
                          df_temp["Địa chỉ"].str.contains(pattern, case=False, na=False)]
    
    if not df_temp.empty:
        df_f = df_temp
    else:
        if property_type:
            pattern = "|".join(map(re.escape, property_type))
            df_f = df_f[df_f["Tên"].str.contains(pattern, case=False, na=False) |
                        df_f["Địa chỉ"].str.contains(pattern, case=False, na=False)]
        
        if df_f.empty: return [] 


    num_features_f = df_f[["Giá", "Số phòng ngủ"]].astype(float)
    if num_features_f.empty: return []

    num_scaled_f = scaler.transform(num_features_f)
    cat_encoded_f = ohe.transform(df_f[["district_reduced"]])
    X_f = np.hstack([num_scaled_f, cat_encoded_f])

    num_input_bedrooms = bedrooms[0] if isinstance(bedrooms, list) and bedrooms else bedrooms
    
    num_input = [
        Giá if Giá is not None else df["Giá"].median(), 
        num_input_bedrooms if num_input_bedrooms is not None else df["Số phòng ngủ"].median()
    ]
    num_scaled_partial = [(num_input[0] - scaler.mean_[0]) / np.sqrt(scaler.var_[0]),
                          (num_input[1] - scaler.mean_[1]) / np.sqrt(scaler.var_[1])]

    if districts and len(districts) > 0:
        d_reduced = districts[0] if districts[0] in top_districts else "OTHER"
        cat_vec = ohe.transform([[d_reduced]]).flatten()
    else:
        cat_vec = np.zeros(ohe.categories_[0].shape[0])

    xq = np.hstack([num_scaled_partial, cat_vec])

    nbrs_f = NearestNeighbors(n_neighbors=min(top_k * 3, len(df_f)), metric="euclidean")
    nbrs_f.fit(X_f)
    distances, indices = nbrs_f.kneighbors([xq])
    max_dist = distances.max() if distances.max() > 0 else 1

    rows = []
    for dist, i in zip(distances.flatten(), indices.flatten()):
        row = df_f.iloc[i].copy()
        row["similarity"] = round(1 - dist / max_dist, 3)
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df = results_df[results_df["similarity"] > 0.3].sort_values("similarity", ascending=False).head(top_k)
    if results_df.empty: return []

    return [{
        "Tên": r.get("Tên"),
        "Địa chỉ": r.get("Địa chỉ"),
        "Số phòng ngủ": int(r.get("Số phòng ngủ", 0)),
        "Giá": float(r.get("Giá", 0)),
        "Gia_fmt": format_gia(r["Giá"]),
        "district": r.get("district"),
        "image_path": r.get("image_path"),
        "URL": r.get("URL"),
        "similarity": float(r.get("similarity", 0.0))
    } for _, r in results_df.iterrows()]

# ======================
# 6) Flask App 
# ======================
app = Flask(__name__)
latest_results = [] 

@app.route("/hinhanh/<path:filename>")
def serve_image(filename):
    abs_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), IMG_FOLDER)
    return send_from_directory(abs_folder, filename.split("/")[-1]) 

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    global latest_results
    data = request.get_json(force=True)
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"ok": False, "reply": " Vui lòng nhập yêu cầu tìm kiếm."})

    crit = parse_query(msg)
    if crit["districts"]:
        crit["districts"] = [d.title().strip() for d in crit["districts"]]

    has_price = crit.get("price_min") is not None or crit.get("price_max") is not None
    broad_search = bool(crit.get("districts")) and (not has_price) 

    if broad_search:
        no_accent_districts = [remove_accents(d).lower().replace(" ", "") for d in crit["districts"]]
        df_f = df[df["district_no_accent"].apply(lambda x: any(d in x for d in no_accent_districts))].copy()
        
        if crit.get("bedrooms") and isinstance(crit["bedrooms"], list) and len(crit["bedrooms"]) > 0:
            df_f = df_f[df_f["Số phòng ngủ"].isin(crit["bedrooms"])]
            
        if crit.get("property_type"):
            pattern = "|".join(map(re.escape, crit["property_type"]))
            df_f = df_f[df_f["Tên"].str.contains(pattern, case=False, na=False) | df_f["Địa chỉ"].str.contains(pattern, case=False, na=False)]

        recs = [{
            "Tên": r.get("Tên"),
            "Địa chỉ": r.get("Địa chỉ"),
            "Số phòng ngủ": int(r.get("Số phòng ngủ", 0)),
            "Giá": float(r.get("Giá", 0)),
            "Gia_fmt": format_gia(r["Giá"]),
            "district": r.get("district"),
            "image_path": r.get("image_path"),
            "URL": r.get("URL"),
            "similarity": 1.0
        } for _, r in df_f.iterrows()]
    else:
        recs = recommend_by_features(Giá=crit["price_max"], price_min=crit["price_min"],
                                     districts=crit["districts"], bedrooms=crit["bedrooms"],
                                     property_type=crit["property_type"], top_k=6)
                                     
    latest_results = recs 
    if not recs:
        return jsonify({"ok": True, "reply": "Không tìm thấy kết quả phù hợp.", "results": []})
    return jsonify({"ok": True, "reply": f"Tìm thấy {len(recs)} căn phù hợp!", "results": recs})

# Chatbot
@app.route("/chat", methods=["POST"])
def chat():
    global latest_results, user_preferences, conversation_history
    user_preferences = globals().get("user_preferences", {})
    conversation_history = globals().get("conversation_history", [])

    data = request.get_json(force=True)
    msg = (data.get("message") or "").strip()
    force_mode = (data.get("force_mode") or "").strip().lower()

    if not msg:
        return jsonify({"ok": False, "reply": "Bạn muốn mình phân tích thêm căn nào không, hay tám chuyện chút cũng được nè 🙂"})

    lower_msg = msg.lower()

    # Intent detection keywords
    GREETING = {"chào", "chào bạn", "hello", "hi", "alo", "yo", "hey", "xin chào", "hola"}
    THANKS = {"cảm ơn", "thanks", "thank you", "cám ơn", "ty"}
    SMALLTALK_PATTERNS = ("khỏe không", "ổn không", "làm gì đó", "đang làm gì", "buồn quá", "vui quá", "ở đó không")
    RE_KEYWORDS = {
        "căn", "căn hộ", "chung cư", "nhà", "biệt thự", "studio", "officetel",
        "quận", "phường", "khu", "view", "giá", "giá bán", "giá thuê",
        "m2", "diện tích", "phòng ngủ", "vị trí", "tiện ích", "pháp lý",
        "đầu tư", "cho thuê", "sinh lời", "hạ tầng", "thanh khoản", "chủ đầu tư"
    }
    PREF_KEYWORDS = ["gần", "trung tâm", "yên tĩnh", "cao", "view", "đầu tư",
                     "cho thuê", "rộng", "rẻ", "cao cấp", "an ninh", "tiện ích", "trường học"]
    CHOICE_KEYWORDS = ["chọn", "đáng mua", "tốt nhất", "nên mua", "tư vấn", "1 căn"]

    if any(w in lower_msg for w in PREF_KEYWORDS):
        user_preferences["ưu tiên"] = msg

    def detect_intent(text: str) -> str:
        t = text.strip().lower()
        if force_mode in {"smalltalk", "real_estate"}:
            return force_mode
        if t in GREETING or any(p in t for p in THANKS) or any(p in t for p in SMALLTALK_PATTERNS):
            return "smalltalk"
        if any(k in t for k in RE_KEYWORDS) or (latest_results and any(k in t for k in CHOICE_KEYWORDS)):
            return "real_estate"

        if len(t.split()) <= 3:
            return "smalltalk"
        return "real_estate"

    intent = detect_intent(lower_msg)
    
    # ⭐️ LOGIC: TÌM CĂN ĐƯỢC CHỌN VÀ TẠO CHỌN PHẢN HỒI
    chosen_apartments_data = None
    crit = parse_query(msg)
    num_to_recommend = crit.get("num_to_recommend", 1)
    
    is_choice_query = latest_results and any(k in lower_msg for k in CHOICE_KEYWORDS)
    
    if is_choice_query and latest_results:
        # Áp dụng bộ lọc giá nếu có
        filtered_results = latest_results.copy()
        
        if "tỷ" in lower_msg or "ty" in lower_msg:
            # Tìm mức giá trong câu hỏi
            price_patterns = [
                # Pattern khoảng giá "từ X đến Y" hoặc "từ 20 đến 30"
                (r"(?:tu|từ)\s*([\d\.]+)\s*(?:den|đến)\s*([\d\.]+)\s*(?:ty|tỷ)", "range"),
                # Pattern giá tối thiểu
                (r"(?:tren|hon|lon hon|từ|tu|toi thieu|toi thieu la|min)\s*([\d\.]+)\s*(?:ty|tỷ)", "min"),
                # Pattern giá tối đa
                (r"(?:duoi|it hon|nho hon|toi da|max)\s*([\d\.]+)\s*(?:ty|tỷ)", "max"),
                # Pattern giá xấp xỉ
                (r"(?:khoang|tam|tầm|khoảng|gia|giá)\s*([\d\.]+)\s*(?:ty|tỷ)", "around")
            ]
            
            for pattern, mode in price_patterns:
                m = re.search(pattern, remove_accents(lower_msg))
                if m:
                    if mode == "range":
                        # Khoảng giá từ X đến Y
                        price_min_val = float(m.group(1)) * 1_000_000_000
                        price_max_val = float(m.group(2)) * 1_000_000_000
                        filtered_results = [r for r in filtered_results 
                                         if float(r.get("Giá", 0)) >= price_min_val 
                                         and float(r.get("Giá", 0)) <= price_max_val]
                    elif mode == "min":
                        price_value = float(m.group(1)) * 1_000_000_000
                        filtered_results = [r for r in filtered_results if float(r.get("Giá", 0)) >= price_value]
                    elif mode == "max":
                        price_value = float(m.group(1)) * 1_000_000_000
                        filtered_results = [r for r in filtered_results if float(r.get("Giá", 0)) <= price_value]
                    elif mode == "around":
                        price_value = float(m.group(1)) * 1_000_000_000
                        margin = price_value * 0.15  # Biên độ 15%
                        filtered_results = [r for r in filtered_results 
                                         if float(r.get("Giá", 0)) >= (price_value - margin)
                                         and float(r.get("Giá", 0)) <= (price_value + margin)]
                    break
        
        # Sắp xếp lại theo similarity để lấy các căn phù hợp nhất sau khi lọc
        filtered_results.sort(key=lambda x: float(x.get("similarity", 0)), reverse=True)
        chosen_apartments_data = filtered_results[:num_to_recommend] if filtered_results else None
        
    # Xây ngữ cảnh
    context_text = ""
    
    # Nếu có chosen_apartments_data (căn được gợi ý), chỉ gửi thông tin của những căn này
    if chosen_apartments_data:
        context_text += f"\n=== THÔNG TIN CHI TIẾT CÁC CĂN HỘ ĐỀ XUẤT ({len(chosen_apartments_data)} căn) ===\n"
        for i, c in enumerate(chosen_apartments_data):
            price = float(c.get('Giá', 0))
            rooms = int(c.get('Số phòng ngủ', 0))
            estimated_area = {1: 45, 2: 65, 3: 85, 4: 100}.get(rooms, 55)
            price_per_m2 = price / estimated_area if estimated_area > 0 else 0
            
            context_text += f"""
Căn {i+1}:
- Tên: {c.get('Tên')}
- Địa chỉ: {c.get('Địa chỉ')}
- Quận: {c.get('district')}
- Giá: {c.get('Gia_fmt')}
- Số phòng ngủ: {rooms}
- Diện tích ước tính: {estimated_area}m²
- Giá/m²: {format_gia(price_per_m2)}/m²
- Độ tương đồng: {round(float(c.get('similarity', 0)) * 100)}%
- Link chi tiết: {c.get('URL', 'Không có')}
"""
        context_text += "=========================================\n"
    # Nếu không có chosen_apartments_data, vẫn hiển thị danh sách tất cả từ latest_results
    elif latest_results:
        top_list = "\n".join([
            f"- {r.get('Tên','?')} ({r.get('district','?')}, {r.get('Gia_fmt','?')}, {r.get('Số phòng ngủ','?')} phòng ngủ) - Similarity: {r.get('similarity', 0.0)}"
            for r in latest_results
        ])
        context_text += f"Các căn hộ người dùng vừa xem (từ top đến thấp):\n{top_list}\n\n"
        
    if user_preferences:
        pref_str = ", ".join([f"{k}: {v}" for k, v in user_preferences.items()])
        context_text += f"Điều kiện & mong muốn người dùng: {pref_str}\n"


    conversation_history.append({"role": "user", "content": msg})
    conversation_history = conversation_history[-6:]
    globals()["conversation_history"] = conversation_history
    globals()["user_preferences"] = user_preferences

    if intent == "real_estate":
        
        prompt = f"""
Bạn là **chuyên gia tư vấn bất động sản cao cấp tại TP.HCM**, với hơn 15 năm kinh nghiệm phân tích và định giá bất động sản. Hãy phân tích một cách chuyên nghiệp và khách quan nhất.

Ngữ cảnh:
{context_text}
Câu hỏi người dùng: {msg}

🏡 Phân tích các căn hộ:
1. So sánh tổng quan:
   - Giá/m² có hợp lý không?
   - Vị trí trong quận có thuận lợi không?
   - Tiện ích xung quanh (điểm mạnh/yếu)

2. Xếp hạng & Đề xuất:
   - Chấm điểm theo: Giá trị (40%) + Vị trí (40%) + Tiện ích (20%)
   - So sánh ưu/nhược điểm chính của từng căn
   - KHÔNG chỉ dựa vào similarity để xếp hạng

3. ✅ Kết luận:
   - Căn hộ được đề xuất 
   - 2 lý do chính cho đề xuất
   - Lời khuyên khi đàm phán (nếu có)

Yêu cầu:
- Phân tích ngắn gọn nhưng đầy đủ tất cả các căn
- Cân nhắc cả giá/m², vị trí và tiện ích
- Đưa ra đề xuất khách quan, có căn cứ"""
        reply = call_gemini(prompt).strip()
        
        if reply and not reply.endswith(("🙂", "😊", "✨", "🌟")):
            reply += "\n\nMuốn mình so thêm vài lựa chọn tương tự không nè? 😊"
            
        # TRẢ VỀ DANH SÁCH CĂN HỘ ĐƯỢC CHỌN
        response_data = {
            "ok": True, 
            "reply": reply, 
            "chosen_apartment_info": chosen_apartments_data 
        }
        
        return jsonify(response_data)

    # SMALL TALK
    hist_text = "\n".join([f"{t['role']:}: {t['content']:}" for t in conversation_history])
    prompt_smalltalk = f"""
Bạn là **trợ lý nói chuyện tự nhiên, dễ thương, ấm áp kiểu Sài Gòn**.
Trả lời 1 đoạn 80–120 từ, đồng cảm, vui vẻ; 0–2 emoji; không nói sang BĐS trừ khi người dùng hỏi.

Lịch sử gần đây:
{hist_text}
Tin nhắn mới: {msg}
"""
    reply = call_gemini(prompt_smalltalk)
    if not reply or not reply.strip():
        reply = "Tớ đây nè! Hôm nay của bạn thế nào rồi rồi? 😊"
    return jsonify({"ok": True, "reply": reply, "chosen_apartment_info": None})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)









