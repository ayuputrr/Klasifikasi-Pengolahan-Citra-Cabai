# Dataset Citra Cabai

## Deskripsi Dataset

Dataset ini berisi citra cabai yang dikategorikan berdasarkan tingkat kematangannya. Dataset dirancang untuk keperluan klasifikasi tingkat kematangan cabai menggunakan teknik pengolahan citra dan machine learning.

## Struktur Dataset

Dataset terdiri dari 3 kelas utama:

| Kelas | Jumlah Citra | Deskripsi |
|-------|--------------|-----------|
| **Matang** | 53 | Citra cabai yang sudah matang sepenuhnya |
| **Belum Matang** | 33 | Citra cabai yang masih dalam tahap pertumbuhan |
| **Kematangan** | 42 | Citra cabai dalam tahap peralihan menuju matang |

**Total Dataset:** 128 citra

## Distribusi Kelas

```
Matang          : 53 citra (41.4%)
Kematangan      : 42 citra (32.8%)
Belum Matang    : 33 citra (25.8%)
```

## Penggunaan Dataset

Dataset ini dapat digunakan untuk:
- Pelatihan model klasifikasi tingkat kematangan cabai
- Penelitian di bidang computer vision untuk pertanian
- Pengembangan sistem otomatis pendeteksi kematangan cabai
- Studi kasus pembelajaran mesin untuk klasifikasi multi-kelas

## Format Data

- **Format File:** JPG/PNG (sesuaikan dengan format citra Anda)
- **Struktur Folder:**
  ```
  dataset/
  ├── matang/          (53 citra)
  ├── belum_matang/    (33 citra)
  └── kematangan/      (42 citra)
  ```

## Catatan Penggunaan

1. Dataset memiliki distribusi yang relatif seimbang dengan kelas mayoritas (Matang) sebesar 41.4%
2. Disarankan untuk melakukan augmentasi data pada kelas minoritas (Belum Matang) untuk meningkatkan performa model
3. Lakukan split data dengan proporsi yang tepat (misalnya 80:20 untuk training:testing)


---
**Terakhir diperbarui:** November 2025
