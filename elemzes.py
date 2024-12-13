# Importáljuk a szükséges könyvtárakat
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import pearsonr

# 1. SEED BEÁLLÍTÁSA
# Seed az adatgenerálás reprodukálhatóságához
SEED = 216114

np.random.seed(SEED)

#Átalakítás ellenőrizzen minden levelet hozzon létre minden levélhez egy összeslevél.count-1 listát minden levelet számozzon meg és mindet ossza be a legjobbhoz


# 2. KULCSSZAVAK INICIALIZÁLÁSA
KULCSSZAVAK = [
    'akció', 'ingyen', 'kedvezmény', 'ajándék', 'limitált', 'exkluzív', 
    'kupon', 'vásárlás', 'hűségprogram', 'garancia', 'megtakarítás', 
    'kedvezményes', 'prémium', 'ajánlat', 'rendelés', 'akciós', 'csomag', 
    'kiváló', 'siker', 'meghívás', 'promóció', 'előfizetés', 'feliratkozás',
    'próbaverzió', 'kedvenc', 'válogatás', 'top', 'szuper', 'csillag'
]

SPAM_SZAVAK = [
    'nyeremény', 'garantált', 'kattints', 'próbálja', 'nyerj', 'akár', 
    'extra', 'sürgős', 'azonnal', 'fontos', 'kérjük', 'visszaigazolás', 
    'biztonságos', 'töltse le', 'ne hagyja ki', 'csatlakozzon', 
    'kérdésed van', 'ingyenes hozzáférés', 'személyre szabott', 'hitelesítés', 
    'regisztráció', 'számla', 'ellenőrzés', 'kód', 'ajánló', 'képzés', 
    'azonnali válasz', 'automatikus üzenet', 'nagy kedvezmény', 
    'támogatás', 'garancia', 'korlátozott'
]

# 3. ADATGENERÁLÓ FÜGGVÉNY (HELYESÍRÁSI HIBÁK HOZZÁADVA)
def generalj_email_adatokat(mintak_szama=30):
    """
    E-mail adatok generálása véletlenszerű paraméterekkel.
    :param mintak_szama: Az e-mailek száma, amelyeket generálunk.
    :return: DataFrame az e-mailek adataival.
    """
    adatok = []
    for _ in range(mintak_szama):
        hossz = np.random.randint(50, 500)
        kulcsszavak_szama = np.random.randint(0, 10)
        spam_szavak_szama = np.random.randint(0, 5)
        cimzettek_szama = np.random.randint(1, 10)
        van_html = np.random.choice([0, 1])
        linkek_szama = np.random.randint(0, 10)
        spec_karakterek_szama = np.random.randint(0, 20)
        formazasok_szama = np.random.randint(0, 5)
        van_melleklet = np.random.choice([0, 1])
        spam_e = np.random.choice([0, 1])

        # Helyesírási hibák száma (több hiba nem spam esetén)
        helyesirasi_hibak = np.random.randint(0, 10) if spam_e == 0 else np.random.randint(0, 3)

        kulcsszo_suruseg = kulcsszavak_szama / hossz if hossz > 0 else 0
        kulso_link_arany = np.random.uniform(0, 1) if linkek_szama > 0 else 0
        spam_szo_arany = spam_szavak_szama / hossz if hossz > 0 else 0

        adatok.append([ 
            hossz, kulcsszavak_szama, spam_szavak_szama, cimzettek_szama, van_html,
            linkek_szama, spec_karakterek_szama, formazasok_szama, van_melleklet,
            helyesirasi_hibak, kulcsszo_suruseg, kulso_link_arany, spam_szo_arany, spam_e
        ])
    
    oszlopok = [
        'hossz',
        'kulcsszavak_szama',
        'spam_szavak_szama',
        'cimzettek_szama',
        'van_html',
        'linkek_szama',
        'spec_karakterek_szama',
        'formazasok_szama',
        'van_melleklet',
        'helyesirasi_hibak', 
        'kulcsszo_suruseg',
        'kulso_link_arany',
        'spam_szo_arany',
        'spam_e'
    ]
    return pd.DataFrame(adatok, columns=oszlopok)

# Generálás 30 mintából
adatok = generalj_email_adatokat(30-2)

adatok_manual = [
    # Nem spam példa
    [
        200,  # hossz
        3,  # kulcsszavak_szama
        0,  # spam_szavak_szama
        5,  # cimzettek_szama
        1,  # van_html (1 - van html, 0 - nincs html)
        1,  # linkek_szama
        0,  # spec_karakterek_szama
        1,  # formazasok_szama
        0,  # van_melleklet
        1,  # helyesirasi_hibak
        0.015,  # kulcsszo_suruseg
        0.01,  # kulso_link_arany
        0.0,  # spam_szo_arany
        0  # spam_e (0 - nem spam)
    ],
    
    # Spam 
    [
        250,  # hossz
        5,  # kulcsszavak_szama
        4,  # spam_szavak_szama
        3,  # cimzettek_szama
        1,  # van_html
        3,  # linkek_szama
        1,  # spec_karakterek_szama
        2,  # formazasok_szama
        1,  # van_melleklet
        2,  # helyesirasi_hibak
        0.02,  # kulcsszo_suruseg
        0.03,  # kulso_link_arany
        0.02,  # spam_szo_arany
        1  # spam_e (1 - spam)
    ]
]

oszlopok = [
    'hossz', 'kulcsszavak_szama', 'spam_szavak_szama', 'cimzettek_szama', 
    'van_html', 'linkek_szama', 'spec_karakterek_szama', 'formazasok_szama',
    'van_melleklet', 'helyesirasi_hibak', 'kulcsszo_suruseg', 'kulso_link_arany', 
    'spam_szo_arany', 'spam_e'
]

adatok_manual_df = pd.DataFrame(adatok_manual, columns=oszlopok)


adatok_full = pd.concat([adatok, adatok_manual_df], ignore_index=True)

# Skála alkalmazása és adatok szétválasztása
skala = MinMaxScaler()
X_full = skala.fit_transform(adatok_full.iloc[:, :-1])
y_full = adatok_full['spam_e']

# Tanító és teszt adatok szétválasztása
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_full, test_size=0.25, random_state=SEED)

# Naiv Bayes modell újratanítása
modell_full = MultinomialNB()
modell_full.fit(X_train_full, y_train_full)

# Predikció és kiértékelés
y_pred_full = modell_full.predict(X_test_full)
print("Osztályozási Jelentés:")
print(classification_report(y_test_full, y_pred_full))
print("Konfúziós Mátrix:")
print(confusion_matrix(y_test_full, y_pred_full))

# 8. KORRELÁCIÓSZÁMÍTÁS
korrelaciok = []
for oszlop in adatok_full.columns[:-1]:
    corr, _ = pearsonr(adatok_full[oszlop], adatok_full['spam_e'])
    korrelaciok.append((oszlop, corr))

korrelaciok = sorted(korrelaciok, key=lambda x: abs(x[1]), reverse=True)
print("\nJellemzők korrelációi a spam osztályozással:")
for jellemzo, corr in korrelaciok:
    print(f"{jellemzo}: {corr:.2f}")

# 9. SPAM ÉS NEM SPAM ARÁNYOK KISZÁMÍTÁSA
spam_db = adatok_full['spam_e'].sum()
nem_spam_db = len(adatok_full) - spam_db
print(f"\nSpam => {spam_db}/{len(adatok_full)} (levelek száma)")
print(f"Nem spam => {nem_spam_db}/{len(adatok_full)} (levelek száma)")


def pontosssagSzamitas(y_actual, y_pred):
    helyes_predikciok = sum(y_actual == y_pred)
    pontossag = helyes_predikciok / len(y_actual) * 100
    print(f"\nModell pontossága: {pontossag:.2f}%")

pontosssagSzamitas(y_test_full, y_pred_full)
