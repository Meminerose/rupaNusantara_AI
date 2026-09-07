import torch

# Sesuaikan dengan nama file model etnis efficientnetb3 kamu
model_path = "checkpoints/model_efficientnetb3_p7_BEST.pth" 

print(f"Mengecek isi file: {model_path}...\n")

try:
    # Memuat file .pth langsung ke CPU agar tidak berat/error memori
    checkpoint = torch.load(model_path, map_location="cpu")
    
    print(f"Tipe data objek di dalam .pth: {type(checkpoint)}\n")
    
    if isinstance(checkpoint, dict):
        # Ambil 5 keys pertama aja biar terminal nggak penuh kalau isinya ribuan layer
        keys = list(checkpoint.keys())
        print(f"Total keys di dalam file: {len(keys)}")
        print("Contoh 5 keys pertama:")
        for k in keys[:5]:
            print(f" - {k}")
            
        print("-" * 40)
        
        # Cek apakah ada informasi mapping tersimpan
        if 'class_to_idx' in checkpoint:
            print("🎉 Hore! Ada mapping kelas di dalamnya:")
            print(checkpoint['class_to_idx'])
        elif 'classes' in checkpoint:
            print("🎉 Hore! Ada daftar kelas di dalamnya:")
            print(checkpoint['classes'])
        elif 'id_to_etnis' in checkpoint:
            print("🎉 Hore! Ada id_to_etnis di dalamnya:")
            print(checkpoint['id_to_etnis'])
        else:
            print("⚠️ Sepertinya ini murni state_dict (hanya berisi bobot / nama-nama layer).")
            print("Artinya, informasi nama etnis TIDAK DISIMPAN di dalam file .pth ini.")
            
    else:
        print("Model bukan berupa dictionary (mungkin di-save secara utuh).")

except Exception as e:
    print(f"Terjadi error saat membaca model: {e}")