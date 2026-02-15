# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 18:29:31 2025

@author: icewa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Algoritmalar
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. VERİ YÜKLEME VE TEMİZLİK (Senin Orijinal Kısmın)
veriseti = pd.read_excel('kaza_verilerr.xlsx', 'traffic_accidents')
veriseti = veriseti.replace({'Y': 1, 'N': 0, 'y': 1, 'n': 0})

cols_to_drop = ['crash_date', 'road_defect', 'crash_type', 'damage', 'injuries_total', 
                'injuries_fatal', 'injuries_incapacitating', 'injuries_non_incapacitating', 
                'injuries_reported_not_evident', 'injuries_no_indication', 'crash_month']
veriseti = veriseti.drop(columns=[c for c in cols_to_drop if c in veriseti.columns])

X = veriseti.drop("most_severe_injury", axis=1)
y = veriseti["most_severe_injury"]

# 2. ENCODING
X = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns, drop_first=False)
le = LabelEncoder()
y = le.fit_transform(y)

# 3. VERİ DENGELEME (Upsampling)
df_egitim = pd.concat([X, pd.Series(y, name='target')], axis=1)
df_majority = df_egitim[df_egitim.target == 2]
df_minority_0 = df_egitim[df_egitim.target == 0]

df_minority_0_upsampled = resample(df_minority_0, replace=True, n_samples=3000, random_state=42)
df_final = pd.concat([df_majority, df_minority_0_upsampled] + [df_egitim[df_egitim.target == i] for i in [1, 3, 4]])

X_resampled = df_final.drop('target', axis=1)
y_resampled = df_final['target']

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# YSA için veriyi ölçeklendirmemiz lazım (Ağaçlar için fark etmez ama YSA için şart)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. MODELLERİ TANIMLAMA VE EĞİTME
# ==========================================
modeller = {
    "Karar Ağacı": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=25, n_jobs=-1, random_state=42),
    "Yapay Sinir Ağları": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

basari_skorlari = {}

print("🚀 Modeller Yarıştırılıyor...")
for isim, model in modeller.items():
    # YSA ise ölçekli veriyi kullan, değilse normali
    if isim == "Yapay Sinir Ağları":
        model.fit(X_train_scaled, y_train)
        tahmin = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        tahmin = model.predict(X_test)
    
    acc = accuracy_score(y_test, tahmin) * 100
    basari_skorlari[isim] = acc
    print(f"✅ {isim} tamamlandı. Başarı: %{acc:.2f}")

# ==========================================
# 5. SONUÇLARI GRAFİĞE DÖKME (Rapora Ekle)
# ==========================================
plt.figure(figsize=(10, 6))
sns.barplot(x=list(basari_skorlari.keys()), y=list(basari_skorlari.values()), palette='viridis')
plt.title('Modellerin Doğruluk Oranı (Accuracy) Karşılaştırması', fontsize=14)
plt.ylabel('Başarı Yüzdesi (%)')
plt.ylim(0, 105)

for i, v in enumerate(basari_skorlari.values()):
    plt.text(i, v + 1, f"%{v:.2f}", ha='center', fontweight='bold')

plt.show()

# En iyi modelin (Random Forest) detaylı raporunu yine basalım
print("\n" + "="*30)
print("EN İYİ MODEL (RANDOM FOREST) DETAYLI RAPORU")
print("="*30)
rf_final = modeller["Random Forest"]
y_pred_rf = rf_final.predict(X_test)
print(classification_report(y_test, y_pred_rf))

# 1. LabelEncoder'ı yeniden kuruyoruz (Sonuçları metne çevirmek için)
# (Orijinal kodda bir değişkene atanmadığı için tekrar fit ediyoruz)
le_tahmin = LabelEncoder()
le_tahmin.fit(veriseti["most_severe_injury"])

# 2. Kullanıcıdan istenecek sütunları belirliyoruz
# (Hedef değişken hariç kalan orijinal sütunlar)
girdi_sutunlari = [col for col in veriseti.columns if col != "most_severe_injury"]

def kaza_tahmini_yap():
    print("\n" + "="*40)
    print("   YENİ KAZA TAHMİN SİSTEMİ")
    print("="*40)
    print("Girilecek inputlar her bir girdi için şunlardan biri olmalı ve büyük harfle yazılmalı: ")    
    print("""         
==================== KULLANICI GİRDİ BİLGİLERİ ====================

1) Trafik Kontrolü (traffic_control_device)
- NO CONTROLS (Kontrol Yok - En Yaygın)
- TRAFFIC SIGNAL (Trafik Işığı)
- STOP SIGN/FLASHER (Dur Tabelası)
- YIELD (Yol Ver)
- LANE USE MARKING (Şerit İşaretleri)
- RAILROAD CROSSING GATE (Tren Geçidi Kapısı)
- SCHOOL ZONE (Okul Bölgesi)
- POLICE/FLAGMAN (Polis/İşaretçi)
- OTHER (Diğer)
- UNKNOWN (Bilinmiyor)

2) Kaza Tipi (first_crash_type)
- REAR END (Arkadan Çarpma)
- TURNING (Dönüşte Çarpışma)
- PARKED MOTOR VEHICLE (Park Halindeki Araca Çarpma)
- FIXED OBJECT (Sabit Cisme Çarpma)
- PEDESTRIAN (Yayaya Çarpma)
- SIDESWIPE SAME DIRECTION (Aynı Yönde Sürtünme)
- ANGLE (Açılı Çarpışma)
- HEAD ON (Kafa Kafaya)
- ANIMAL (Hayvana Çarpma)
- PEDALCYCLIST (Bisikletliye Çarpma)

3) Hava Durumu (weather_condition)
- CLEAR (Açık)
- RAIN (Yağmurlu)
- SNOW (Karlı)
- CLOUDY/OVERCAST (Bulutlu/Kapalı)
- FOG/SMOKE/HAZE (Sisli/Puslu)
- SLEET/HAIL (Dolu)
- FREEZING RAIN/DRIZZLE (Dondurucu Yağmur)
- BLOWING SNOW (Tipi)

4) Işıklandırma (lighting_condition)
- DAYLIGHT (Gündüz)
- DARKNESS (Karanlık - Işık Yok)
- DARKNESS, LIGHTED ROAD (Karanlık - Sokak Lambalı)
- DUSK (Alacakaranlık)
- DAWN (Şafak)

5) Ana Sebep (prim_contributory_cause)
- UNABLE TO DETERMINE (Belirlenemedi)
- FAILING TO YIELD RIGHT-OF-WAY (Yol Hakkı Vermeme)
- FOLLOWING TOO CLOSELY (Yakın Takip)
- IMPROPER OVERTAKING/PASSING (Hatalı Sollama)
- FAILING TO REDUCE SPEED (Hız Kesmemek)
- IMPROPER TURNING/NO SIGNAL (Hatalı Dönüş)
- DRIVING SKILLS/KNOWLEDGE/EXPERIENCE (Sürücü Acemiliği)
- WEATHER (Hava Şartları)
- DISREGARDING TRAFFIC SIGNALS (Işık İhlali)
- DISREGARDING STOP SIGN (Dur Tabelası İhlali)
- OPERATING VEHICLE IN ERRATIC MANNER (Tehlikeli Sürüş)
- PHYSICAL CONDITION OF DRIVER (Sürücünün Fiziksel Durumu)
- DISTRACTION - FROM INSIDE VEHICLE (Araç İçi Dikkat Dağınıklığı)
- EQUIPMENT - VEHICLE CONDITION (Araç Arızası)

6) Yol Tipi (trafficway_type)
- NOT DIVIDED (Bölünmemiş Yol)
- DIVIDED - W/MEDIAN (Refüjlü Bölünmüş Yol)
- ONE-WAY (Tek Yön)
- FOUR WAY (Dört Yol Ağzı)
- T-INTERSECTION (T Kavşak)
- PARKING LOT (Otopark)
- ROUNDABOUT (Döner Kavşak)
- RAMP (Rampa/Bağlantı Yolu)

7) Yol Yüzeyi (roadway_surface_cond)
- DRY (Kuru)
- WET (Islak)
- SNOW OR SLUSH (Karlı/Erimiş Kar)
- ICE (Buzlu)
- SAND, MUD, DIRT (Kum/Çamur)

8) Yol Eğimi (alignment)
- STRAIGHT AND LEVEL (Düz ve Eğimsi̇z)
- CURVE, LEVEL (Virajlı ama Düz)
- STRAIGHT ON GRADE (Eğimli Düz Yol)
- CURVE ON GRADE (Eğimli Viraj)

9) Araç Sayısı (num_units)
- 1 ile 10 arasında bir tam sayı giriniz
  Örnek: 1, 2, 5, 10

10) Haftanın Günü (day_of_week)
- 1 ile 7 arasında bir tam sayı giriniz
  1 = Pazartesi
  2 = Salı
  3 = Çarşamba
  4 = Perşembe
  5 = Cuma
  6 = Cumartesi
  7 = Pazar

11) Kaza Saati (crash_hour)
- 1 ile 24 arasında bir tam sayı giriniz

12) Kesişim bölgesinde mi ? (intersection_related_i)
    1 ya da 0 giriniz.

Lütfen girdileri ingilizce ve büyük harfle olacak şekilde gösterildiği gibi giriniz.

==================================================================
""")
    user_input = {}
    
    for col in girdi_sutunlari:
        deger = input(f"{col}: ")
        
        # Sayısal veri kontrolü: Verisetindeki sütun tipine bakıyoruz
        dtype = veriseti[col].dtype
        if np.issubdtype(dtype, np.number):
            try:
                user_input[col] = float(deger)
            except ValueError:
                print(f"UYARI: Sayısal değer bekleniyordu. 0.0 atandı.")
                user_input[col] = 0.0
        else:
            # Metin verileri için büyük harfe çevir (Model eğitimiyle uyum için)
            user_input[col] = deger.upper()
    
    # 3. Veriyi DataFrame'e çeviriyoruz
    df_user = pd.DataFrame([user_input])
    # 4. Model formatına uygun hale getirme (En Kritik Adım)
    # Kullanıcının girdiği veriyi dummy değişkenlere çevirir
    df_user = pd.get_dummies(df_user)
    # Eğitim setindeki (X) sütun yapısını buraya zorluyoruz.
    # Eksik sütunları 0 ile doldurur, fazla varsa atar.
    df_user = df_user.reindex(columns=X.columns, fill_value=0)
    # 5. Tahmin
    tahmin_indeks = model.predict(df_user)
    tahmin_sonuc = le_tahmin.inverse_transform(tahmin_indeks)
    print("\n" + "-"*30)
    print(f"TAHMİN EDİLEN KAZA SONUCU: {tahmin_sonuc[0]}")
    print("-"*30)
# Fonksiyonu çalıştır
kaza_tahmini_yap()