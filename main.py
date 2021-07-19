
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = yf.download(['AAPL', 'GE', 'BAC', 'AMD', 'PLUG', 'F'], period='3mo')
closeData = data.Close  # дневные курсы закрытия
dCloseData = closeData.pct_change()  # относительные изменения к предыдущему дню
dohMean = dCloseData.mean()  # Средняя дневная доходность по каждой акции
cov = dCloseData.cov()  # ковариационная матрица
for name in closeData.columns:  # Графики курсов
    closeData[name].plot()  # Графики курсов
    plt.grid()  # Графики курсов
    plt.title(name)  # Графики курсов
    plt.show()  # Графики курсов

cnt = len(dCloseData.columns)  # Генерация рандмного портфеля


# Генерация рандмного портфеля
def randPortf():  # Генерация рандмного портфеля
    res = np.exp(np.random.randn(cnt))  # Генерация рандмного портфеля
    res = res / res.sum()  # Генерация рандмного портфеля
    return res  # Генерация рандмного портфеля


# Генерация рандмного портфеля
r = randPortf()  # Генерация рандмного портфеля
print(r)  # Генерация рандмного портфеля
print(r.sum())  # Генерация рандмного портфеля


def dohPortf(r):  # Доходность портфеля
    return np.matmul(dohMean.values, r)  # Доходность портфеля


# Доходность портфеля
r = randPortf()  # Доходность портфеля
print(r)  # Доходность портфеля
d = dohPortf(r)  # Доходность портфеля
print(d)  # Доходность портфеля


def riskPortf(r):  # Риск портфеля
    return np.sqrt(np.matmul(np.matmul(r, cov.values), r))  # Риск портфеля


# Риск портфеля
r = randPortf()  # Риск портфеля
print(r)  # Риск портфеля
rs = riskPortf(r)  # Риск портфеля
print(rs)  # Риск портфеля

# Облако портфелей и График риск доходность
N = 1000
risk = np.zeros(N)
doh = np.zeros(N)
portf = np.zeros((N, cnt))

for n in range(N):
    r = randPortf()

    portf[n, :] = r
    risk[n] = riskPortf(r)
    doh[n] = dohPortf(r)

plt.figure(figsize=(10, 8))

plt.scatter(risk * 100, doh * 100, c='y', marker='.')
plt.xlabel('риск, %')
plt.ylabel('доходность, %')
plt.title("Облако портфелей")

min_risk = np.argmin(risk)
plt.scatter([(risk[min_risk]) * 100], [(doh[min_risk]) * 100], c='r', marker='*', label='минимальный риск')

maxSharpKoef = np.argmax(doh / risk)
plt.scatter([risk[maxSharpKoef] * 100], [doh[maxSharpKoef] * 100], c='g', marker='o', label='максимальный коэф-т Шарпа')

r_mean = np.ones(cnt) / cnt
risk_mean = riskPortf(r_mean)
doh_mean = dohPortf(r_mean)
plt.scatter([risk_mean * 100], [doh_mean * 100], c='b', marker='x', label='усредненный портфель')

plt.legend()

plt.show()
#Вывод данных в таблицу
print('---------- Минимальный риск ----------')
print()
print("риск = %1.2f%%" % (float(risk[min_risk]) * 100.))
print("доходность = %1.2f%%" % (float(doh[min_risk]) * 100.))
print()
print(pd.DataFrame([portf[min_risk] * 100], columns=dCloseData.columns, index=['доли, %']).T)
print()

print('---------- Максимальный коэффициент Шарпа ----------')
print()
print("риск = %1.2f%%" % (float(risk[maxSharpKoef]) * 100.))
print("доходность = %1.2f%%" % (float(doh[maxSharpKoef]) * 100.))
print()
print(pd.DataFrame([portf[maxSharpKoef] * 100], columns=dCloseData.columns, index=['доли, %']).T)
print()

print('---------- Средний портфель ----------')
print()
print("риск = %1.2f%%" % (float(risk_mean) * 100.))
print("доходность = %1.2f%%" % (float(doh_mean) * 100.))
print()
print(pd.DataFrame([r_mean * 100], columns=dCloseData.columns, index=['доли, %']).T)
print()
