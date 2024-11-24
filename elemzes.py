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
SEED = 1
np.random.seed(SEED)

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
        'helyesirasi_hibak',  # Helyesírási hibák oszlop
        'kulcsszo_suruseg',
        'kulso_link_arany',
        'spam_szo_arany',
        'spam_e'
    ]
    return pd.DataFrame(adatok, columns=oszlopok)

# Generálás 30 mintából
adatok = generalj_email_adatokat(30)

# 4. ADATSIMÍTÁS
skala = MinMaxScaler()
X = skala.fit_transform(adatok.iloc[:, :-1])
y = adatok['spam_e']

# 5. TANÍTÓ ÉS TESZT ADATOK SZÉTVÁLASZTÁSA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

# 6. NAIV BAYES MODELL TANÍTÁSA
modell = MultinomialNB()
modell.fit(X_train, y_train)

# 7. PREDIKCIÓ ÉS KIÉRTÉKELÉS
y_pred = modell.predict(X_test)
print("Osztályozási Jelentés:")
print(classification_report(y_test, y_pred))
print("Konfúziós Mátrix:")
print(confusion_matrix(y_test, y_pred))

# 8. KORRELÁCIÓSZÁMÍTÁS
korrelaciok = []
for oszlop in adatok.columns[:-1]:
    corr, _ = pearsonr(adatok[oszlop], adatok['spam_e'])
    korrelaciok.append((oszlop, corr))

korrelaciok = sorted(korrelaciok, key=lambda x: abs(x[1]), reverse=True)
print("\nJellemzők korrelációi a spam osztályozással:")
for jellemzo, corr in korrelaciok:
    print(f"{jellemzo}: {corr:.2f}")

# 9. SPAM ÉS NEM SPAM ARÁNYOK KISZÁMÍTÁSA
spam_db = adatok['spam_e'].sum()
nem_spam_db = len(adatok) - spam_db
print(f"\nSpam => {spam_db}/{len(adatok)} (levelek száma)")
print(f"Nem spam => {nem_spam_db}/{len(adatok)} (levelek száma)")
