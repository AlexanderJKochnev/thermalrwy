import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Параметры ячейки
CELL_MASS = 0.045  # кг (вес INR18750-50 ~45 г)
CELL_SURFACE_AREA = 0.007  # м² (приблизительно)
CELL_SPECIFIC_HEAT = 900  # Дж/(кг·К) — усреднённое значение литий-ионных ячеек

# Теплофизические параметры окружающей среды
T_AMBIENT = 25 + 273.15  # К (температура окружающей среды)
H_CONV = 10  # Вт/(м²·К), коэффициент конвекции
EMISSIVITY = 0.9  # степень черноты
SIGMA = 5.67e-8  # Стефана–Больцмана

# Параметры тепловыделения при TR
TR_ENERGY_NCA = 5000  # Дж
TR_ENERGY_LFP = 2500  # Дж
TR_DURATION = 10  # секунд (пиковая фаза)

def q_generation(t, chemistry='NCA'):
    """Моделирование тепловыделения в зависимости от времени и химии"""
    if t < 100:
        return 0.0  # индукционная фаза
    elif 100 <= t < 200:
        return 0.01 * (t - 100) * (TR_ENERGY_NCA if chemistry == 'NCA' else TR_ENERGY_LFP) / TR_DURATION
    elif 200 <= t < 300:
        return (TR_ENERGY_NCA if chemistry == 'NCA' else TR_ENERGY_LFP) / TR_DURATION
    else:
        return max(0, ((300 - t) / 100) * (TR_ENERGY_NCA if chemistry == 'NCA' else TR_ENERGY_LFP) / TR_DURATION)

def heat_loss(T):
    """Рассчёт теплопотерь через конвекцию и излучение"""
    q_conv = H_CONV * CELL_SURFACE_AREA * (T - T_AMBIENT)
    q_rad = EMISSIVITY * SIGMA * CELL_SURFACE_AREA * (T**4 - T_AMBIENT**4)
    return q_conv + q_rad

def model(t, T, chemistry='NCA'):
    q_gen = q_generation(t, chemistry)
    q_loss = heat_loss(T)
    dTdt = (q_gen - q_loss) / (CELL_MASS * CELL_SPECIFIC_HEAT)
    return dTdt

# Временной диапазон
t_span = [0, 600]  # 10 минут
t_eval = np.linspace(0, 600, 1000)

# Решение дифференциального уравнения
sol_nca = solve_ivp(model, t_span, [T_AMBIENT], t_eval=t_eval, args=('NCA',))
sol_lfp = solve_ivp(model, t_span, [T_AMBIENT], t_eval=t_eval, args=('LFP',))

# Перевод температуры в °C
temps_nca = sol_nca.y[0] - 273.15
temps_lfp = sol_lfp.y[0] - 273.15

# Расчёт тепловыделения и потерь
q_gens = np.array([q_generation(t, 'NCA') for t in t_eval])
q_losses = np.array([heat_loss(sol_nca.y[0][i]) for i, t in enumerate(t_eval)])

# === Графики ===
plt.figure(figsize=(12, 6))

# Температура
plt.subplot(2, 1, 1)
plt.plot(t_eval, temps_nca, label="NCA", color='red')
plt.plot(t_eval, temps_lfp, label="LFP", color='blue')
plt.title("Температура ячейки при тепловом разгоне")
plt.ylabel("Температура, °C")
plt.grid(True)
plt.legend()

# Тепловыделение и потери
plt.subplot(2, 1, 2)
plt.plot(t_eval, q_gens, label="Тепловыделение", color='orange')
plt.plot(t_eval, q_losses, label="Теплопотери", color='green')
plt.title("Мощность тепловыделения и теплопотери")
plt.xlabel("Время, с")
plt.ylabel("Мощность, Вт")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()