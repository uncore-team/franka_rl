import pandas as pd
import matplotlib.pyplot as plt

def plot_reward(csv: str, figname: str, percent: float):
        
    # Leer el CSV
    df = pd.read_csv(csv)

    # Extraer columnas
    steps = df["Step"].values
    values = df["Value"].values
    times = df["Wall time"].values

    # Calcular mínimo y máximo de recompensa
    val_min = min(values)
    val_max = max(values)

    # Calcular umbral del 95% del rango
    threshold = val_min + percent * (val_max - val_min)

    # Buscar tiempo de establecimiento
    settling_index = None
    for i in range(len(values)):
        if all(values[j] >= threshold for j in range(i, len(values))):
            settling_index = i
            break

    if settling_index is not None:
        settling_step = steps[settling_index]
        settling_time = times[settling_index]
        settling_value = values[settling_index]
        print(f"Tiempo de establecimiento: Paso {settling_step}, Tiempo {settling_time}, Valor {settling_value}")
    else:
        print("No se encontró un tiempo de establecimiento dentro del 95%.")

    # Graficar
    plt.plot(steps, values, label="Recompensa")
    plt.axhline(threshold, color='red', linestyle='--', label='Umbral 95% del valor final')
    if settling_index is not None:
        plt.axvline(settling_step, color='green', linestyle='--', label='Establecido')
        # Punto y texto
        plt.scatter(settling_step, settling_value, color='green', zorder=5)
        plt.text(settling_step+1000, settling_value-2, f"({settling_step},{settling_value:.2f})", color='black',
                ha='left', va='bottom', fontsize=12)
    plt.xlabel("Paso")
    plt.ylabel("Recompensa")
    plt.legend()
    plt.savefig(figname, format="pdf")
    plt.show()

if __name__=="__main__":
    # CONFIGURATION
    csv = "reach4/reach4.csv"
    figname = "reach4/reward" # save the resulting figure with this name
    percent = 0.95 # [0,1]
    ####################################
    plot_reward(csv=csv, figname=figname, percent=percent)
