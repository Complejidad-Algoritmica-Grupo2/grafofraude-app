"""
Sistema de Detección de Fraude basado en Grafos
Escenario 2: Tarjeta -> Transacciones
Algoritmo: Programación Dinámica (estilo Kadane) sobre patrones temporales

Idea central: no toda la secuencia de transacciones de una tarjeta es
sospechosa. Buscamos el TRAMO con el riesgo acumulado más alto, en vez
de evaluar transacciones de forma aislada.

Dataset esperado (CSV): columnas mínimas requeridas
    TransactionID, TransactionDT, TransactionAmt,
    card1, card2, card3, card4, card5, card6

NOTA DE REVISIÓN: este archivo conserva la lógica original de Winnie
sin modificaciones. Los comentarios fueron agregados únicamente con
fines de comprensión/documentación (sin alterar el comportamiento).
"""

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# - Definición de variables globales
# ---------------------------------------------------------------------------
# Ruta relativa al CSV de transacciones (desde donde se ejecuta el programa).
RUTA_TRANSACTION = "fraud_detector_program/data/train_transaction.csv"

# Mismo tamaño de muestra que Escenario 1, para coherencia entre escenarios.
N_REGISTROS = 1500

# Si el delta de tiempo entre dos transacciones consecutivas de la MISMA
# tarjeta es <= 60 unidades, se considera una "ráfaga" sospechosa.
# (TransactionDT es un delta relativo, no segundos reales del reloj.)
UMBRAL_DT_RAFAGA = 60

# Monto a partir del cual una transacción se considera "alto riesgo por monto".
UMBRAL_MONTO_ALTO = 500

# Cuántas tarjetas del top mostrar en las conclusiones.
TOP_TARJETAS_A_MOSTRAR = 5

# ---------------------------------------------------------------------------
# CARGA Y PREPARACIÓN DE DATOS
# ---------------------------------------------------------------------------
def cargar_datos(ruta: str, n_muestra: int = N_REGISTROS) -> pd.DataFrame:
    """
    Carga el dataset de transacciones y construye el nodo proxy 'Tarjeta'
    a partir de card1-card6. Toma una muestra de n_muestra registros.

    Args:
        ruta (str): ruta al CSV de transacciones.
        n_muestra (int): cantidad máxima de registros a usar (default 1500).

    Returns:
        pd.DataFrame: dataframe muestreado con la columna "Tarjeta" agregada.

    Raises:
        ValueError: si el CSV no contiene alguna de las columnas necesarias.
    """
   
    # Solo se cargan las columnas que este escenario necesita.
    # card1-card6 son atributos enmascarados de la tarjeta; juntos
    # forman un identificador proxy más robusto que card1 solo.
    columnas = [
        "TransactionID", "TransactionDT", "TransactionAmt",
        "card1", "card2", "card3", "card4", "card5", "card6"
    ]
    df = pd.read_csv(ruta, usecols=columnas)

    # Verificación defensiva antes de continuar.
    faltantes = [c for c in columnas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el dataset: {faltantes}")

    # min() asegura que no se pidan más filas de las que existen.
    n = min(n_muestra, len(df))
    df = df.sample(n=n, random_state=42).reset_index(drop=True)
    print(f"Muestra tomada: {n} registros (de {len(df)} disponibles)")

    # Rellenar nulos para que la concatenación de strings no genere "nan"
    # dentro del identificador proxy de tarjeta.
    df = df.fillna({
        "card1": "NA", "card2": "NA", "card3": "NA",
        "card4": "NA", "card5": "NA", "card6": "NA",
        "TransactionAmt": 0
    })

    # Nodo proxy "Tarjeta": se concatenan los 6 campos de tarjeta.
    # Como el dataset los tiene enmascarados, esta combinación es el
    # identificador más completo posible para distinguir una tarjeta de otra.
    df["Tarjeta"] = (
        df["card1"].astype(str) + "_" + df["card2"].astype(str) + "_" +
        df["card3"].astype(str) + "_" + df["card4"].astype(str) + "_" +
        df["card5"].astype(str) + "_" + df["card6"].astype(str)
    )

    return df


# ---------------------------------------------------------------------------
# SCORE DE RIESGO POR TRANSACCIÓN
# ---------------------------------------------------------------------------
def calcular_score_riesgo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna un score de riesgo a cada transacción, dentro de su tarjeta,
    en función de:
      - Monto alto (+1): TransactionAmt >= UMBRAL_MONTO_ALTO
      - Ráfaga (+1): poco tiempo desde la transacción anterior de la MISMA tarjeta
      - Comportamiento normal (-0.5): costo base que evita que la DP acumule
        ruido indefinidamente — solo un tramo genuinamente sospechoso
        logra superar esta penalización.

    Args:
        df (pd.DataFrame): dataframe cargado con cargar_datos().

    Returns:
        pd.DataFrame: mismo dataframe con las columnas "delta_t" y "score" agregadas.
    """
    # Ordenar por tarjeta y luego por tiempo, para que el diff() siguiente
    # calcule correctamente la diferencia respecto a la transacción ANTERIOR
    # de la MISMA tarjeta (no de otra tarjeta diferente).
    df = df.sort_values(["Tarjeta", "TransactionDT"]).reset_index(drop=True)

    # groupby().diff() calcula la diferencia entre cada valor y el anterior
    # DENTRO del mismo grupo (Tarjeta). La primera transacción de cada
    # tarjeta queda como NaN (no tiene transacción previa con la que comparar).
    df["delta_t"] = df.groupby("Tarjeta")["TransactionDT"].diff()

    def score_fila(row):
        """Calcula el score de riesgo de una sola transacción.

        Args:
            row: fila del dataframe (Series de pandas).

        Returns:
            float: score de riesgo de esa transacción.
        """
        # Empieza en -0.5: si no hay ningún indicador sospechoso,
        # la transacción "drena" el riesgo acumulado — efecto central
        # de la variante Kadane aplicada aquí: un solo evento sospechoso
        # aislado no hace subir el score total si el resto son normales.
        score = -0.5
        if row["TransactionAmt"] >= UMBRAL_MONTO_ALTO:
            score += 1.0   # monto alto: señal de posible cargo fraudulento grande
        if pd.notna(row["delta_t"]) and row["delta_t"] <= UMBRAL_DT_RAFAGA:
            score += 1.0   # ráfaga temporal: muchas transacciones en poco tiempo
        return score

    # apply(axis=1) aplica score_fila a cada fila del dataframe.
    df["score"] = df.apply(score_fila, axis=1)
    return df


# ---------------------------------------------------------------------------
# PROGRAMACIÓN DINÁMICA: máxima subsecuencia de riesgo (estilo Kadane)
# ---------------------------------------------------------------------------
def dp_max_subsecuencia_riesgo(scores: list) -> tuple:
    """
    Encuentra el TRAMO CONTIGUO de mayor riesgo acumulado en la lista
    de scores usando el algoritmo de Kadane adaptado.

    Algoritmo de Kadane original: máxima subarray sum en O(n).
    Aquí se extiende para rastrear también los índices de inicio y fin
    del tramo óptimo, no solo el valor acumulado.

    Transición de la DP:
        dp[i] = max(score[i], dp[i-1] + score[i])
        Si dp[i-1] + score[i] > score[i] → conviene extender el tramo anterior.
        Si no → conviene iniciar un tramo nuevo desde i.

    Args:
        scores (list[float]): lista de scores de riesgo ordenados cronológicamente.

    Returns:
        tuple: (riesgo_maximo, indice_inicio, indice_fin)
            - riesgo_maximo (float): acumulado del tramo más sospechoso.
            - indice_inicio (int): posición donde arranca ese tramo.
            - indice_fin (int): posición donde termina ese tramo.
            Devuelve (0.0, -1, -1) si la lista está vacía.
    """
    if not scores:
        return 0.0, -1, -1

    # Inicializar con el primer elemento (el tramo mínimo es 1 transacción).
    dp_actual = scores[0]    # acumulado del tramo que termina en la posición actual
    max_riesgo = scores[0]   # mejor acumulado visto hasta ahora
    inicio_actual = 0        # dónde empezó el tramo actual
    mejor_inicio, mejor_fin = 0, 0  # índices del mejor tramo encontrado

    for i in range(1, len(scores)):
        if dp_actual + scores[i] > scores[i]:
            # Extender el tramo actual es mejor que empezar uno nuevo.
            dp_actual = dp_actual + scores[i]
        else:
            # Empezar un tramo nuevo desde i es mejor (el acumulado anterior
            # era negativo y arrastraba el score hacia abajo).
            dp_actual = scores[i]
            inicio_actual = i

        # Si el acumulado actual supera el mejor visto, actualizar el resultado.
        if dp_actual > max_riesgo:
            max_riesgo = dp_actual
            mejor_inicio = inicio_actual
            mejor_fin = i

    return max_riesgo, mejor_inicio, mejor_fin


def analizar_todas_las_tarjetas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la DP a cada tarjeta por separado y devuelve un resumen
    con el tramo de mayor riesgo encontrado para cada una.

    Args:
        df (pd.DataFrame): dataframe con columnas "Tarjeta", "score",
            "TransactionDT" y "TransactionAmt".

    Returns:
        pd.DataFrame: una fila por tarjeta, ordenada de mayor a menor
            riesgo acumulado, con columnas descriptivas del tramo detectado.
    """
    resultados = []

    # groupby("Tarjeta") agrupa las transacciones por tarjeta proxy.
    # Cada "grupo" es el historial completo de una tarjeta en la muestra.
    for tarjeta, grupo in df.groupby("Tarjeta"):
        grupo = grupo.sort_values("TransactionDT").reset_index(drop=True)
        scores = grupo["score"].tolist()

        # Con solo 1 transacción no hay "tramo" que analizar — se omite.
        if len(scores) < 2:
            continue

        riesgo_max, i_inicio, i_fin = dp_max_subsecuencia_riesgo(scores)

        resultados.append({
            "Tarjeta": tarjeta,
            "riesgo_max": riesgo_max,
            # Cuántas transacciones conforman el tramo sospechoso detectado.
            "n_transacciones_tramo": i_fin - i_inicio + 1,
            "n_transacciones_total": len(grupo),
            "dt_inicio": grupo.loc[i_inicio, "TransactionDT"],
            "dt_fin": grupo.loc[i_fin, "TransactionDT"],
            # Monto total de dinero involucrado en el tramo sospechoso.
            "monto_total_tramo": grupo.loc[i_inicio:i_fin, "TransactionAmt"].sum(),
        })

    # sort_values deja las tarjetas más sospechosas primero.
    return pd.DataFrame(resultados).sort_values("riesgo_max", ascending=False)


# ---------------------------------------------------------------------------
# VISUALIZACIÓN: serie de tiempo de riesgo acumulado para una tarjeta
# ---------------------------------------------------------------------------
def graficar_tarjeta(df: pd.DataFrame, tarjeta: str, guardar_como: str = None):
    """
    Grafica la serie de transacciones de una tarjeta, marcando el tramo
    de mayor riesgo detectado por la DP.

    Args:
        df (pd.DataFrame): dataframe completo con columna "Tarjeta".
        tarjeta (str): identificador proxy de la tarjeta a graficar.
        guardar_como (str, opcional): si se indica, guarda el PNG en esa
            ruta en vez de abrir ventana interactiva con plt.show().
    """
    # Filtrar solo las transacciones de la tarjeta indicada y ordenarlas.
    grupo = df[df["Tarjeta"] == tarjeta].sort_values("TransactionDT").reset_index(drop=True)
    scores = grupo["score"].tolist()

    # Recalcular el tramo óptimo para esta tarjeta específica.
    riesgo_max, i_inicio, i_fin = dp_max_subsecuencia_riesgo(scores)

    plt.figure(figsize=(12, 5))

    # Línea gris: el score de cada transacción individual.
    plt.plot(grupo.index, grupo["score"], marker="o", color="gray", label="Score por transacción")

    # axvspan dibuja un rectángulo semitransparente sobre el tramo sospechoso,
    # los ±0.3 son solo márgenes visuales para que el rectángulo no quede
    # exactamente en el borde del marcador de cada punto.
    plt.axvspan(i_inicio - 0.3, i_fin + 0.3, color="red", alpha=0.2,
                label=f"Tramo de mayor riesgo (acumulado={riesgo_max:.1f})")

    plt.title(f"Tarjeta: {tarjeta}\nTramo sospechoso: transacciones {i_inicio} a {i_fin}",
              fontsize=12, fontweight="bold")
    plt.xlabel("Índice de transacción (orden cronológico)")
    plt.ylabel("Score de riesgo")

    # Línea horizontal en 0 para visualizar fácilmente qué transacciones
    # contribuyen positivamente al riesgo y cuáles lo reducen.
    plt.axhline(0, color="black", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    if guardar_como:
        plt.savefig(guardar_como, dpi=150, bbox_inches="tight")
        print(f"Gráfico guardado en: {guardar_como}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CONCLUSIONES
# ---------------------------------------------------------------------------
def generar_conclusiones(resumen: pd.DataFrame):
    """Imprime en consola un resumen interpretativo de los resultados del escenario.

    Args:
        resumen (pd.DataFrame): dataframe devuelto por analizar_todas_las_tarjetas().
    """
    print("\n" + "=" * 70)
    print("CONCLUSIONES - Escenario 2: Tarjeta -> Transacciones")
    print("=" * 70)

    print(f"\nTotal de tarjetas analizadas (con >= 2 transacciones): {len(resumen)}")

    print(f"\nTop {TOP_TARJETAS_A_MOSTRAR} tarjetas con mayor riesgo acumulado:")
    for _, fila in resumen.head(TOP_TARJETAS_A_MOSTRAR).iterrows():
        print(f"\n  Tarjeta: {fila['Tarjeta']}")
        print(f"    Riesgo acumulado máximo: {fila['riesgo_max']:.2f}")
        print(f"    Transacciones en el tramo sospechoso: {fila['n_transacciones_tramo']} "
              f"(de {fila['n_transacciones_total']} totales)")
        print(f"    Ventana temporal del tramo: DT {fila['dt_inicio']:.0f} -> {fila['dt_fin']:.0f}")
        print(f"    Monto total en el tramo: {fila['monto_total_tramo']:.2f}")

    print("\nInterpretación:")
    print("  - Un riesgo acumulado alto concentrado en pocas transacciones")
    print("    sugiere una RÁFAGA puntual (posible prueba de tarjeta robada).")
    print("  - Un riesgo acumulado alto distribuido en muchas transacciones")
    print("    sugiere actividad sostenida anómala en esa tarjeta.")


# ---------------------------------------------------------------------------
# FLUJO PRINCIPAL DEL ESCENARIO
# ---------------------------------------------------------------------------
def ejecutar_escenario_2():
    """Orquesta el flujo completo del Escenario 2: carga, cálculo de scores,
    DP por tarjeta, conclusiones y generación del gráfico de la tarjeta top."""
    print("\nCargando dataset...")
    df = cargar_datos(RUTA_TRANSACTION)

    print("\nCalculando score de riesgo por transacción...")
    df = calcular_score_riesgo(df)

    print("\nEjecutando Programación Dinámica por tarjeta (máxima subsecuencia de riesgo)...")
    resumen = analizar_todas_las_tarjetas(df)

    generar_conclusiones(resumen)

    if len(resumen) > 0:
        # Tomar la tarjeta con mayor riesgo (ya viene ordenado desc).
        tarjeta_top = resumen.iloc[0]["Tarjeta"]
        print(f"\nGenerando gráfico de la tarjeta con mayor riesgo: {tarjeta_top}...")
        graficar_tarjeta(df, tarjeta_top, guardar_como="riesgo_escenario2.png")


# Permite ejecutar este escenario directamente con:
# python algorithm/scenario2.py
if __name__ == "__main__":
    ejecutar_escenario_2()