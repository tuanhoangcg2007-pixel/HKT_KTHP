import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import unicodedata
import os
import urllib.request
from PIL import Image

# --- CẤU HÌNH GIAO DIỆN WEB ---
st.set_page_config(page_title="Canteen Thông Minh", page_icon="🍽️", layout="centered")
st.title("🍽️ HỆ THỐNG TÍNH TIỀN KHAY CƠM TỰ ĐỘNG")
st.write("Hệ thống nhận diện món ăn bằng CNN & Quét QR chuyển khoản nhanh")

# --- KHỞI TẠO BỘ NÃO CNN (Tự động tải từ Google Drive của bạn nếu chưa có) ---
@st.cache_resource
def load_my_model():
    model_filename = "MOHINHHKTSIEUCAPVIPPRO.keras"
    GOOGLE_DRIVE_ID = "1r8zDDrXfsGmXQzTnAotnSH4vR62LOEt5" 
    direct_download_url = f"https://docs.google.com/uc?export=download&id={GOOGLE_DRIVE_ID}"
    
    if not os.path.exists(model_filename):
        with st.spinner("📦 Đang tải bộ não CNN từ Google Drive về máy chủ Streamlit (File nặng, chỉ tải duy nhất lần đầu tiên, vui lòng đợi từ 1-2 phút)..."):
            try:
                urllib.request.urlretrieve(direct_download_url, model_filename)
                st.success("📥 Đã tải xong bộ não!")
            except Exception as e:
                st.error(f"❌ Lỗi khi tải mô hình từ Drive: {e}")
                st.stop()
            
    return tf.keras.models.load_model(model_filename)

try:
    model = load_my_model()
    st.success("🧠 Bộ não CNN đã nạp thành công! Sẵn sàng nhận diện.")
except Exception as e:
    st.error(f"❌ Lỗi nạp mô hình: {e}")
    st.stop()

# --- DATA CẤU HÌNH (DANH SÁCH MÓN, GIÁ, TỌA ĐỘ) ---
CLASS_NAMES = [
    'Cơm trắng', 'Đậu hũ sốt cà', 'Cá hú kho', 'Thịt kho trứng', 'Thịt kho',
    'Canh chua có cá', 'Canh chua không cá', 'Sườn nướng', 'Canh rau', 'Rau xào', 'Trứng chiên'
]

def quick_norm(text):
    return unicodedata.normalize('NFC', text).lower().strip()

MENU_PRICES = {
    quick_norm('Cơm trắng'): 10000, quick_norm('Đậu hũ sốt cà'): 25000,
    quick_norm('Cá hú kho'): 30000, quick_norm('Thịt kho trứng'): 30000,
    quick_norm('Thịt kho'): 25000, quick_norm('Canh chua có cá'): 25000,
    quick_norm('Canh chua không cá'): 10000, quick_norm('Sườn nướng'): 30000,
    quick_norm('Canh rau'): 7000, quick_norm('Rau xào'): 10000, quick_norm('Trứng chiên'): 25000
}

BOXES = [
    {"X": 389, "Y": 182, "W": 513, "H": 419},   # Ngăn 1
    {"X": 1088, "Y": 182, "W": 398, "H": 419},  # Ngăn 2
    {"X": 397, "Y": 669, "W": 329, "H": 324},   # Ngăn 3
    {"X": 794, "Y": 669, "W": 335, "H": 324},   # Ngăn 4
    {"X": 1159, "Y": 669, "W": 369, "H": 324}   # Ngăn 5
]


# =================================================================
# 📸 PHẦN CHỌN PHƯƠNG THỨC ĐƯA ẢNH VÀO: TẢI FILE HOẶC CHỤP TRỰC TIẾP
# =================================================================
st.subheader("📸 Đưa dữ liệu khay cơm vào hệ thống")

# Tạo 2 Tab đẹp đẽ nằm ngang trên giao diện Web
tab1, tab2 = st.tabs(["📁 Tải ảnh từ máy tính (Chèn file)", "📷 Chụp trực tiếp bằng Camera"])

img_file = None

with tab1:
    uploaded_file = st.file_uploader("Chọn file ảnh khay cơm của bạn (.jpg, .png, .jpeg):", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img_file = uploaded_file

with tab2:
    camera_file = st.camera_input("Hãy đưa khay cơm vào chính giữa khung hình và bấm chụp:")
    if camera_file is not None:
        img_file = camera_file


# =================================================================
# 🧠 XỬ LÝ NHẬN DIỆN VÀ TÍNH TIỀN (Sẽ chạy khi có ảnh từ 1 trong 2 nguồn)
# =================================================================
if img_file is not None:
    # Đọc ảnh chuyển sang định dạng dữ liệu byte và đưa về OpenCV
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    
    total_price = 0
    results = []
    
    # Tiến hành vòng lặp cắt ảnh và nhận diện 5 ngăn theo tọa độ cố định
    for i, box in enumerate(BOXES):
        x, y, w, h = box["X"], box["Y"], box["W"], box["H"]
        cropped = img_rgb[y : y + h, x : x + w]
        
        if cropped.size == 0:
            continue
            
        resized = cv2.resize(cropped, (224, 224))
        input_tensor = tf.expand_dims(resized, axis=0)
        
        # Gọi mô hình CNN dự đoán món ăn
        preds = model.predict(input_tensor, verbose=0)
        class_idx = np.argmax(preds[0])
        confidence = np.max(preds[0])
        predicted_food = CLASS_NAMES[class_idx]
        
        # Ngưỡng tin cậy > 65% để loại bỏ khay trống
        if confidence > 0.65:
            norm_name = quick_norm(predicted_food)
            price = MENU_PRICES.get(norm_name, 0)
            total_price += price
            results.append(f"📍 **Ngăn {i+1}:** {predicted_food} ({confidence*100:.1f}%) ➔ **{price:,} VNĐ**")
        else:
            results.append(f"📍 **Ngăn {i+1}:** Khay trống hoặc không rõ món")
            
    # --- HIỂN THỊ HÓA ĐƠN ---
    st.subheader("📋 Chi tiết hóa đơn:")
    for res in results:
        st.write(res)
        
    st.markdown("---")
    st.markdown(f"### 💰 TỔNG TIỀN: <span style='color:red'>{total_price:,} VNĐ</span>", unsafe_allow_html=True)
    
    # --- HIỂN THỊ ẢNH MÃ QR CỐ ĐỊNH CỦA BẠN ---
    if total_price > 0:
        st.subheader("📲 Quét mã QR bên dưới để thanh toán chuyển khoản:")
        
        qr_image_path = "qr_cua_toi.jpg"  # File ảnh QR tĩnh của bạn trên GitHub
        
        if os.path.exists(qr_image_path):
            img_qr = Image.open(qr_image_path)
            st.image(img_qr, caption="Vui lòng nhập đúng số tiền khi chuyển khoản", width=300)
        else:
            st.error(f"⚠️ Không tìm thấy file ảnh '{qr_image_path}' trên thư mục GitHub. Hãy bổ sung file ảnh QR ngân hàng của bạn nhé!")