import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from pybaselines.whittaker import asls

df = pd.read_csv("140.txt", sep=r"\s+", names=["X", "Y"], decimal=',')


n = 100 # число спектров
m = len(df)//n # число точек в одном спектре

# X берем из первого спектра
X = df["X"].iloc[:m].reset_index(drop=True)

# Y превращаем в матрицу (m x n)
Y = df["Y"].values.reshape(n, m).T


result = pd.DataFrame(Y, columns=[f"Y{i+1}" for i in range(n)])
result.insert(0, "X", X)


#функции обработки спектра
def remove_baseline(y, lam = 10^5, p = 0.005):
    baseline, _ = asls(y, lam = lam, p = p)
    return y - baseline

def smooth(y, window=9, polyorder=2):
    return savgol_filter(y, window_length=window, polyorder=polyorder)

#усредненный спектр по всем
processed = pd.DataFrame()
processed['X'] = X

for col in result.columns[1:]:
    y = result[col].values
    y_no_baseline = remove_baseline(y)
    y_smooth = smooth(y_no_baseline)
    processed[col] = y_smooth

mean_spectrum = processed.iloc[:, 1:].mean(axis=1)

#усредненный спектр по 5 лучшим
max_intensities = processed.iloc[:, 1:].max(axis=0)
top5_columns = max_intensities.sort_values(ascending=False).head(5).index
mean_top5 = processed[top5_columns].mean(axis=1)

#4 пика для всех
peaks_all, _ = find_peaks(mean_spectrum, distance = 20)
top4_all = peaks_all[np.argsort(mean_spectrum[peaks_all])[-4:]]

#4 пика для 5
peaks_top5, _ = find_peaks(mean_top5, distance = 20)
top4_top5 = peaks_top5[np.argsort(mean_top5[peaks_top5])[-4:]]


plt.figure(figsize=(8, 5))
plt.plot(X, mean_spectrum, label="Средний по всем спектрам", linewidth=2)
plt.plot(X, mean_top5, label="Средний по 5 самым интенсивным", linewidth=2)


plt.scatter(X.iloc[top4_all], mean_spectrum.iloc[top4_all], zorder=3)
plt.scatter(X.iloc[top4_top5], mean_top5.iloc[top4_top5], zorder=3)

# подписи пиков (все спектры)
for idx in top4_all:
    wn = X.iloc[idx]
    inten = mean_spectrum.iloc[idx]
    plt.annotate(
        f"({wn:.0f}, {inten:.1f})",
        xy=(wn, inten),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )

# подписи пиков (топ-5)
for idx in top4_top5:
    wn = X.iloc[idx]
    inten = mean_top5.iloc[idx]
    plt.annotate(
        f"({wn:.0f}, {inten:.1f})",
        xy=(wn, inten),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )

plt.gca().invert_xaxis()
plt.xlabel("Волновое число, см⁻¹")
plt.ylabel("Интенсивность")
plt.title("Сравнение усреднённых обработанных SERS-спектров")
plt.legend()
plt.tight_layout()
plt.show()
