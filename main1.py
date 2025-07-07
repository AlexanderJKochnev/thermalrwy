import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# === Параметры ===
N_CELLS = 65           # 13s5p = 65 ячеек
CELL_MASS = 0.045      # кг
CELL_SPECIFIC_HEAT = 900  # Дж/(кг·К)
CELL_SURFACE_AREA = 0.007  # м²
T_AMBIENT = 25 + 273.15    # К
H_CONV = 10                # Вт/(м²·К)
EMISSIVITY = 0.9
SIGMA = 5.67e-8

# === Теплопередача между ячейками ===
K_MATERIAL = 200       # теплопроводность материала между ячейками, например алюминий
DISTANCE = 0.02        # м (расстояние между центрами ячеек)
CONTACT_AREA = 0.002   # м²

# === Энергия теплового разгона ===
TR_ENERGY_NCA = 5000   # Дж
TR_DURATION = 10       # секунд

# === Временной диапазон ===
t_span = [0, 600]
t_eval = np.linspace(0, 600, 1000)

# === Тепловыделение для каждой ячейки ===
def q_generation(t, cell_idx, triggered):
    if not triggered[cell_idx]:
        return 0.0
    elif t < 100:
        return 0.0
    elif 100 <= t < 200:
        return 0.01 * (t - 100) * TR_ENERGY_NCA / TR_DURATION
    elif 200 <= t < 300:
        return TR_ENERGY_NCA / TR_DURATION
    else:
        return max(0, ((300 - t) / 100) * TR_ENERGY_NCA / TR_DURATION)

# === Теплопотери ===
def heat_loss(T):
    q_conv = H_CONV * CELL_SURFACE_AREA * (T - T_AMBIENT)
    q_rad = EMISSIVITY * SIGMA * CELL_SURFACE_AREA * (T**4 - T_AMBIENT**4)
    return q_conv + q_rad

# === Модель с распространением ===
def model(t, Ts, triggered):
    dTsdt = np.zeros_like(Ts)
    for i in range(N_CELLS):
        T = Ts[i]

        # Тепловыделение
        q_gen = q_generation(t, i, triggered)

        # Теплопотери
        q_loss = heat_loss(T)

        # Теплопроводность к соседям
        q_cond = 0
        if i > 0:
            q_cond += K_MATERIAL * CONTACT_AREA * (Ts[i - 1] - T) / DISTANCE
        if i < N_CELLS - 1:
            q_cond += K_MATERIAL * CONTACT_AREA * (Ts[i + 1] - T) / DISTANCE

        # Уравнение теплопередачи
        dTsdt[i] = (q_gen + q_cond - q_loss) / (CELL_MASS * CELL_SPECIFIC_HEAT)

        # Активация соседних ячеек при достижении порога TR
        if T >= 150 + 273.15 and not triggered[i]:  # Температура начала TR ~150°C
            triggered[i] = True

    return dTsdt

# === Инициализация ===
initial_temps = np.array([T_AMBIENT] * N_CELLS)
triggered = [False] * N_CELLS
triggered[0] = True  # Запускаем TR в первой ячейке

# === Решение ===
sol = solve_ivp(lambda t, y: model(t, y, triggered), t_span, initial_temps, t_eval=t_eval, rtol=1e-3, atol=1e-3)

# Переводим в °C
temps_celsius = sol.y.T - 273.15

# === Графики ===
plt.figure(figsize=(14, 8))

# === Температура ячеек во времени ===
plt.subplot(2, 1, 1)
for i in range(N_CELLS):
    plt.plot(t_eval, temps_celsius[:, i], color='red', alpha=0.1)
plt.plot(t_eval, temps_celsius[:, 0], color='blue', label="Ячейка 0")
plt.title("Температура ячеек при тепловом разгоне (13S5P)")
plt.ylabel("Температура, °C")
plt.grid(True)
plt.legend()

# === Общая мощность тепловыделения ===
plt.subplot(2, 1, 2)
# total_q_gen = np.sum([q_generation(t, i, triggered) for i in range(N_CELLS)] for t in t_eval), axis=1)
total_q_gen = np.array([
    sum(q_generation(t, i, triggered) for i in range(N_CELLS))
    for t in t_eval
])
plt.plot(t_eval, total_q_gen, color='orange')
plt.title("Общая мощность тепловыделения в модуле")
plt.xlabel("Время, с")
plt.ylabel("Мощность, Вт")
plt.grid(True)

plt.tight_layout()
plt.show()