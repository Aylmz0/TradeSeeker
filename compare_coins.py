import requests

def get_klines(symbol, interval='1h', limit=48):  # 48 saat
    url = f'https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}'
    resp = requests.get(url)
    data = resp.json()
    prices = [float(k[4]) for k in data]
    return prices

def normalize(lst):
    base = lst[0]
    return [(x/base - 1)*100 for x in lst]

def corr(a, b):
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n)) / n
    std_a = (sum((x - mean_a)**2 for x in a) / n) ** 0.5
    std_b = (sum((x - mean_b)**2 for x in b) / n) ** 0.5
    return cov / (std_a * std_b) if std_a > 0 and std_b > 0 else 0

# Coin verileri
coins = {
    'ADA': normalize(get_klines('ADAUSDT')),
    'DOGE': normalize(get_klines('DOGEUSDT')),
    'XRP': normalize(get_klines('XRPUSDT')),
    'LINK': normalize(get_klines('LINKUSDT')),
    'TRX': normalize(get_klines('TRXUSDT')),
    'SOL': normalize(get_klines('SOLUSDT')),
}

result = """
=== MEVCUT COINLER ARASI KORELASYON (48 saat) ===

"""

# Korelasyon matrisi
coin_list = ['ADA', 'DOGE', 'XRP', 'LINK', 'SOL', 'TRX']
result += "        ADA    DOGE   XRP    LINK   SOL    TRX\n"
result += "-" * 55 + "\n"

for c1 in coin_list:
    row = f"{c1:6} "
    for c2 in coin_list:
        if c1 == c2:
            row += " 1.00  "
        else:
            r = corr(coins[c1], coins[c2])
            row += f" {r:.2f}  "
    result += row + "\n"

# Ortalama korelasyon hesapla
ada_corr_with_others = [corr(coins['ADA'], coins[c]) for c in ['DOGE', 'XRP', 'LINK', 'SOL']]
trx_corr_with_others = [corr(coins['TRX'], coins[c]) for c in ['DOGE', 'XRP', 'LINK', 'SOL']]

ada_avg = sum(ada_corr_with_others) / len(ada_corr_with_others)
trx_avg = sum(trx_corr_with_others) / len(trx_corr_with_others)

result += f"""
=== ORTALAMA KORELASYON ===
ADA'nin diger coinlerle ort. korelasyonu: {ada_avg:.2f}
TRX'in diger coinlerle ort. korelasyonu: {trx_avg:.2f}

=== SONUC ===
"""

if trx_avg < ada_avg:
    fark = ada_avg - trx_avg
    result += f"TRX, ADA'ya gore {fark:.2f} puan DAHA BAGIMSIZ hareket ediyor!"
    result += "\nTRX ile degisim portfoy diversifikasyonunu ARTTIRIR."
else:
    result += "ADA ve TRX benzer korelasyon gosteriyor, degisimin etkisi az."

with open('compare_result.txt', 'w') as f:
    f.write(result)

print("Sonuc compare_result.txt dosyasina yazildi")
