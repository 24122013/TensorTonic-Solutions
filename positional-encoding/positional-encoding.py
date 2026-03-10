import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    # 1. Khởi tạo ma trận kết quả toàn số 0
    pe = np.zeros((seq_len, d_model))
    
    # 2. Tạo vector vị trí (pos): shape (seq_len, 1)
    positions = np.arange(seq_len).reshape(-1, 1)
    
    # 3. Tính toán cho các cột CHẴN (sin)
    # Các chỉ số i chẵn: 0, 2, 4... < d_model
    div_term_even = np.power(base, np.arange(0, d_model, 2) / d_model)
    pe[:, 0::2] = np.sin(positions / div_term_even)
    
    # 4. Tính toán cho các cột LẺ (cos)
    # Các chỉ số i lẻ: 1, 3, 5... < d_model
    # Lưu ý: Công thức gốc dùng 2i, nên ở đây ta dùng (i-1) để khớp mẫu số
    div_term_odd = np.power(base, (np.arange(1, d_model, 2) - 1) / d_model)
    pe[:, 1::2] = np.cos(positions / div_term_odd)
    
    return pe