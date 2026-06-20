import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import unicodedata
import os
import gdown

# --- 1. CẤU HÌNH GIAO DIỆN WEB ---
st.set_page_config(page_title="Canteen Tự Động", page_icon="🍽️", layout="centered")
st.title("🍽️ HỆ THỐNG TÍNH TIỀN KHAY CƠM TỰ ĐỘNG")
st.write("Nhận diện món ăn & Quét QR chuyển khoản siêu tốc")

# --- 2. KHỞI TẠO BỘ NÃO CNN (Tải từ Google Drive) ---
@st.cache_resource
def load_my_model():
    model_filename = "MOHINHHKTSIEUCAPVIPPRO.keras"
    # ID file mô hình trên Drive
    GOOGLE_DRIVE_ID = "1r8zDDrXfsGmXQzTnAotnSH4vR62LOEt5" 
    
    if not os.path.exists(model_filename):
        with st.spinner("📦 Đang tải bộ não CNN từ Google Drive (Vui lòng đợi 1-3 phút cho lần chạy đầu tiên)..."):
            try:
                url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
                gdown.download(url, model_filename, quiet=False)
                st.success("📥 Đã tải xong file bộ não!")
            except Exception as e:
                st.error(f"❌ Lỗi khi tải mô hình từ Drive: {e}")
                st.stop()
                
    try:
        return tf.keras.models.load_model(model_filename)
    except Exception as e:
        if os.path.exists(model_filename):
            os.remove(model_filename)
        st.error(f"❌ Lỗi đọc mô hình. Hãy F5 tải lại trang: {e}")
        st.stop()

model = load_my_model()

# --- 3. DỮ LIỆU CẤU HÌNH: TÊN MÓN & GIÁ TIỀN ---
CLASS_NAMES = [
    'Cahukho', 'Canh chua cá', 'Canh chua không cá', 'Canh rau', 
    'Comtrang', 'Dauhusotca', 'Sườn nướng', 'Thitkhotrung', 
    'Thịt kho không trứng', 'rau cu xào', 'trứng chiên'
]

def quick_norm(text):
    return unicodedata.normalize('NFC', text).lower().strip()

MENU_PRICES = {
    quick_norm('Cahukho'): 30000,
    quick_norm('Canh chua cá'): 25000,
    quick_norm('Canh chua không cá'): 10000,
    quick_norm('Canh rau'): 7000,
    quick_norm('Comtrang'): 10000,
    quick_norm('Dauhusotca'): 25000,
    quick_norm('Sườn nướng'): 30000,
    quick_norm('Thitkhotrung'): 30000,
    quick_norm('Thịt kho không trứng'): 25000,
    quick_norm('rau cu xào'): 10000,
    quick_norm('trứng chiên'): 25000
}

# Tọa độ gốc chuẩn của khay cơm (Giả định dựa trên box lớn nhất của bạn là khoảng 1600x1000)
# Hệ thống sẽ tự động scale ảnh đầu vào về kích thước chuẩn này trước khi cắt
TARGET_WIDTH = 1600
TARGET_HEIGHT = 1080

BOXES = [
    {"X": 389, "Y": 182, "W": 513, "H": 419},   
    {"X": 1088, "Y": 182, "W": 398, "H": 419},  
    {"X": 397, "Y": 669, "W": 329, "H": 324},   
    {"X": 794, "Y": 669, "W": 335, "H": 324},   
    {"X": 1159, "Y": 669, "W": 369, "H": 324}   
]

OUTPUT_DIR = "cropped_dishes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 4. GIAO DIỆN CHỌN NGUỒN ẢNH ---
st.subheader("📸 Đưa dữ liệu khay cơm vào hệ thống")
tab1, tab2 = st.tabs(["📁 Tải ảnh từ máy", "📷 Chụp trực tiếp"])

img_file = None
with tab1:
    uploaded_file = st.file_uploader("Chọn ảnh khay cơm (.jpg, .png):", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None: img_file = uploaded_file
with tab2:
    camera_file = st.camera_input("Chụp ảnh khay cơm từ camera:")
    if camera_file is not None: img_file = camera_file

# --- 5. XỬ LÝ NHẬN DIỆN & TÍNH TIỀN ---
if img_file is not None:
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    
    # SỬA LỖI TỌA ĐỘ: Ép kích cỡ ảnh về size chuẩn cấu hình BOXES ban đầu
    img_rgb = cv2.resize(img_rgb, (TARGET_WIDTH, TARGET_HEIGHT))
    
    img_display = img_rgb.copy()
    total_price = 0
    results = []
    
    for i, box in enumerate(BOXES):
        x, y, w, h = box["X"], box["Y"], box["W"], box["H"]
        
        # Kiểm tra bảo vệ chống tràn boundary
        if y + h > TARGET_HEIGHT or x + w > TARGET_WIDTH:
            continue
            
        # Vẽ khung xanh lá lên ảnh hiển thị
        cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(img_display, f"Ngan {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Cắt ảnh ngăn đồ ăn
        cropped = img_rgb[y : y + h, x : x + w]
        if cropped.size == 0: continue
        
        # Lưu ảnh cắt vào folder server
        save_path = os.path.join(OUTPUT_DIR, f"ngan_{i+1}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
            
        # Tiền xử lý cho CNN (Chuẩn hóa /255.0 nếu mô hình của bạn yêu cầu)
        resized = cv2.resize(cropped, (224, 224))
        input_tensor = tf.expand_dims(resized, axis=0)
        input_tensor = tf.cast(input_tensor, tf.float32) # Giúp đồng bộ kiểu dữ liệu cho mô hình
        
        # Dự đoán
        preds = model.predict(input_tensor, verbose=0)
        class_idx = np.argmax(preds[0])
        confidence = np.max(preds[0])
        predicted_food = CLASS_NAMES[class_idx]
        
        if confidence > 0.65:
            norm_name = quick_norm(predicted_food)
            price = MENU_PRICES.get(norm_name, 0)
            total_price += price
            results.append(f"📍 **Ngăn {i+1}:** {predicted_food} ({confidence*100:.1f}%) ➔ **{price:,} VNĐ**")
        else:
            results.append(f"📍 **Ngăn {i+1}:** (Khay trống / Không nhận diện được)")
            
    # --- 6. HIỂN THỊ KẾT QUẢ RA WEB ---
    st.subheader("🔍 Ảnh nhận diện thực tế:")
    st.image(img_display, caption="Hệ thống tự động cắt 5 ngăn và lưu trữ độc lập", use_container_width=True)
    
    st.subheader("📋 Chi tiết hóa đơn:")
    for res in results:
        st.write(res)
        
    st.markdown("---")
    st.markdown(f"### 💰 TỔNG TIỀN: <span style='color:red'>{total_price:,} VNĐ</span>", unsafe_allow_html=True)
    
    if total_price > 0:
        st.subheader("📲 Quét mã QR thanh toán:")
        YOUR_BANK = "mbb" 
        YOUR_ACCOUNT = "0838088267"
        YOUR_NAME = "TRINH HOANG TUAN"
        
        qr_url = f"https://img.vietqr.io/image/{YOUR_BANK}-{YOUR_ACCOUNT}-compact2.jpg?amount={total_price}&addInfo=Tien%20Com%20Canteen&accountName={YOUR_NAME.replace(' ', '%20')}"
        
        st.image(qr_url, caption=f"Chuyển khoản chính xác {total_price:,} VNĐ", width=300)
