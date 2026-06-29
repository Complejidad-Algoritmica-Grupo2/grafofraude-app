"""
Sistema de Detección de Fraude basado en Grafos
Escenario 1: Dispositivo -> Cuenta
Algoritmo: BFS/DFS para componentes conexas + métricas de grado

Dataset esperado (CSV): columnas mínimas requeridas
    TransactionID, TransactionDT, card1, card2, addr1,
    DeviceType, DeviceInfo

NOTA DE REVISIÓN: este archivo conserva la lógica original de Winnie
sin modificaciones. Los comentarios fueron agregados únicamente con
fines de comprensión/documentación (sin alterar el comportamiento).
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from collections import deque

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# - Definir variables globales
# ---------------------------------------------------------------------------
# Rutas donde el programa espera encontrar los CSV del dataset IEEE-CIS.
# Son rutas relativas a la carpeta desde la que se ejecuta el programa.
RUTA_TRANSACTION = "fraud_detector_program/data/train_transaction.csv"
RUTA_IDENTITY = "fraud_detector_program/data/train_identity.csv"

# Umbral de "sospecha": un dispositivo conectado a 5 o más cuentas
# distintas se considera anómalo (ver sección 2.1.1 del informe TF).
UMBRAL_CUENTAS_SOSPECHOSAS = 5        # un dispositivo con >= N cuentas se marca

# ---------------------------------------------------------------------------
# CARGA Y PREPARACIÓN DE DATOS
# ---------------------------------------------------------------------------
def cargar_datos(RUTA_TRANSACTION: str, RUTA_IDENTITY: str) -> pd.DataFrame:
    
    """
    Carga transaction.csv e identity.csv y las combina con un LEFT JOIN
    sobre TransactionID (no todas las transacciones tienen identity).
    Luego construye las columnas proxy de Dispositivo y Cuenta.

    Args:
        RUTA_TRANSACTION (str): ruta al CSV de transacciones.
        RUTA_IDENTITY (str): ruta al CSV de identidad/dispositivo.

    Returns:
        pd.DataFrame: dataframe ya limpio, muestreado (máx. 1500 filas)
        y con las columnas proxy "Cuenta" y "Dispositivo" agregadas.

    Raises:
        ValueError: si después del merge faltan columnas necesarias
        para el resto del pipeline.
    """

    # Solo se leen las columnas que el escenario realmente necesita,
    # evitando cargar las ~370 columnas totales del dataset IEEE-CIS.
    columnas_transaction = ["TransactionID", "TransactionDT", "card1", "card2", "addr1"]
    columnas_identity = ["TransactionID", "DeviceType", "DeviceInfo"]

    df_transaction = pd.read_csv(RUTA_TRANSACTION, usecols=columnas_transaction)
    df_identity = pd.read_csv(RUTA_IDENTITY, usecols=columnas_identity)

    # LEFT JOIN: se conservan todas las transacciones, aunque no tengan
    # fila correspondiente en identity.csv (quedarán con NaN en DeviceType/Info).
    df = df_transaction.merge(df_identity, on="TransactionID", how="left")

    # Opción 1: excluir transacciones sin datos de dispositivo.
    # No tiene sentido analizar conectividad de un nodo "Dispositivo"
    # que en realidad significa "sin información".
    filas_antes = len(df)
    
    # dropna(..., how="all") elimina la fila solo si AMBAS columnas
    # (DeviceType y DeviceInfo) son nulas a la vez.
    df = df.dropna(subset=["DeviceType", "DeviceInfo"], how="all")
    filas_excluidas = filas_antes - len(df)
    print(f"Transacciones sin datos de dispositivo excluidas: {filas_excluidas} de {filas_antes}")

    # Se trabaja solo con una muestra de 1500 registros (alcance del proyecto),
    # tomada YA SOBRE el subset que sí tiene datos de dispositivo.
    # random_state=42 fija la semilla para que la muestra sea reproducible.
    n_muestra = min(1500, len(df))
    df = df.sample(n=n_muestra, random_state=42).reset_index(drop=True)
    print(f"Muestra tomada: {n_muestra} registros (de {filas_antes - filas_excluidas} con dispositivo disponible)")

    # Verificación defensiva: si el merge o el filtrado dejaron el
    # dataframe sin alguna columna esperada, se detiene la ejecución
    # con un mensaje claro en vez de fallar más adelante de forma críptica.
    columnas_necesarias = [
        "TransactionID", "TransactionDT", "card1", "card2",
        "addr1", "DeviceType", "DeviceInfo"
    ]
    faltantes = [c for c in columnas_necesarias if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el dataset: {faltantes}")

    # Reemplazar nulos para que la concatenación de strings (más abajo)
    # no falle ni produzca "nan" como texto dentro del identificador.
    df = df.fillna({
        "card1": "NA", "card2": "NA", "addr1": "NA",
        "DeviceType": "NA", "DeviceInfo": "NA"
    })

    # Nodo proxy "Cuenta": como el dataset no trae un ID de cuenta real,
    # se construye uno combinando tarjeta + dirección (card1_card2_addr1).
    # Dos transacciones con la misma combinación se asumen de la misma cuenta.
    df["Cuenta"] = (
        df["card1"].astype(str) + "_" +
        df["card2"].astype(str) + "_" +
        df["addr1"].astype(str)
    )

    # Nodo "Dispositivo": combinación de tipo de dispositivo + info detallada
    # (ej. "mobile_iOS Device"). Funciona como identificador proxy del device.
    df["Dispositivo"] = (
        df["DeviceType"].astype(str) + "_" + df["DeviceInfo"].astype(str)
    )

    return df


# ---------------------------------------------------------------------------
# CONSTRUCCIÓN DEL GRAFO
# ---------------------------------------------------------------------------
def construir_grafo(df: pd.DataFrame) -> nx.Graph:
    """
    Construye un grafo bipartito Dispositivo-Cuenta.
    Peso de arista = diferencia de TransactionDT entre transacciones
    consecutivas que comparten el mismo par Dispositivo-Cuenta.

    Args:
        df (pd.DataFrame): dataframe ya preparado por cargar_datos(),
            debe contener las columnas "Dispositivo" y "Cuenta".

    Returns:
        nx.Graph: grafo no dirigido con nodos tipados ("Dispositivo"
        o "Cuenta") y aristas con atributos "tiempo" y "frecuencia".
    """
    G = nx.Graph()

    # Ordenar por dispositivo y tiempo para calcular deltas consecutivos.
    # (El ordenamiento se calcula pero el atributo "tiempo" de la arista
    # en realidad solo guarda el primer TransactionDT visto, no un delta
    # real entre transacciones — ver nota de mejora más abajo.)
    df_ordenado = df.sort_values(["Dispositivo", "TransactionDT"])

    for _, fila in df_ordenado.iterrows():
        # Prefijos "DISP::" y "CTA::" evitan colisiones de nombre si un
        # mismo string pudiera representar tanto un dispositivo como una
        # cuenta (namespacing simple dentro del grafo).
        disp = f"DISP::{fila['Dispositivo']}"
        cuenta = f"CTA::{fila['Cuenta']}"

        # add_node es idempotente: si el nodo ya existe, no lo duplica.
        G.add_node(disp, tipo="Dispositivo")
        G.add_node(cuenta, tipo="Cuenta")

        if G.has_edge(disp, cuenta):
            # Si ya existe la arista, acumulamos cuántas veces se repite
            # esta combinación dispositivo-cuenta (proxy de "actividad").
            G[disp][cuenta]["frecuencia"] += 1
        else:
            G.add_edge(disp, cuenta, tiempo=fila["TransactionDT"], frecuencia=1)

    return G


# ---------------------------------------------------------------------------
# DIBUJO EN CONSOLA (ASCII)
# ---------------------------------------------------------------------------
def dibujar_grafo_ascii(G: nx.Graph, max_dispositivos: int = 10):
    """Imprime una lista de adyacencia legible, limitada para no saturar consola.

    Args:
        G (nx.Graph): grafo Dispositivo-Cuenta ya construido.
        max_dispositivos (int): cantidad máxima de dispositivos a
            mostrar antes de truncar la salida (por defecto 10).
    """
    print("\n" + "=" * 70)
    print("GRAFO (vista ASCII) - Dispositivo -> Cuenta")
    print("=" * 70)

    # Filtra solo los nodos cuyo atributo "tipo" es "Dispositivo".
    dispositivos = [n for n, d in G.nodes(data=True) if d["tipo"] == "Dispositivo"]

    for disp in dispositivos[:max_dispositivos]:
        vecinos = list(G.neighbors(disp))
        print(f"\n[{disp}]  (grado={len(vecinos)})")
        for cuenta in vecinos[:10]:
            freq = G[disp][cuenta]["frecuencia"]
            print(f"   └──> {cuenta}   (frecuencia={freq})")
        if len(vecinos) > 10:
            print(f"   ... y {len(vecinos) - 10} cuentas más")

    if len(dispositivos) > max_dispositivos:
        print(f"\n... y {len(dispositivos) - max_dispositivos} dispositivos más (omitidos)")


# ---------------------------------------------------------------------------
# DIBUJO VISUAL (matplotlib) - solo nodos sospechosos, para que sea legible
# ---------------------------------------------------------------------------
def dibujar_grafo_visual(G: nx.Graph, umbral: int, guardar_como: str = None):
    """
    Dibuja el grafo con matplotlib, pero SOLO el subgrafo formado por
    dispositivos sospechosos (grado >= umbral) y sus cuentas conectadas.
    Esto evita que el dibujo sea ilegible con datasets grandes.

    Args:
        G (nx.Graph): grafo Dispositivo-Cuenta completo.
        umbral (int): grado mínimo para considerar un dispositivo sospechoso.
        guardar_como (str, opcional): si se indica, guarda el PNG en esa
            ruta en vez de abrir una ventana interactiva con plt.show().
    """
    dispositivos = [n for n, d in G.nodes(data=True) if d["tipo"] == "Dispositivo"]
    sospechosos = [n for n in dispositivos if G.degree(n) >= umbral]

    if not sospechosos:
        print("\nNo hay dispositivos sospechosos para graficar "
              f"(ningún dispositivo alcanza el umbral >= {umbral}).")
        return

    # Subgrafo: dispositivos sospechosos + todas sus cuentas vecinas.
    # Se usa un set para evitar nodos duplicados al iterar vecinos.
    nodos_subgrafo = set(sospechosos)
    for disp in sospechosos:
        nodos_subgrafo.update(G.neighbors(disp))

    # G.subgraph() devuelve una VISTA (no una copia) del grafo original
    # restringida a esos nodos; es eficiente en memoria.
    SG = G.subgraph(nodos_subgrafo)

    disp_sub = [n for n in sospechosos]
    cta_sub = [n for n in nodos_subgrafo if n.startswith("CTA::")]

    plt.figure(figsize=(14, 9))
    # bipartite_layout posiciona un conjunto de nodos en una columna
    # y el resto en otra, ideal para grafos bipartitos como este.
    pos = nx.bipartite_layout(SG, disp_sub)

    # El tamaño visual del nodo escala con su grado, para que los
    # dispositivos más conectados destaquen visualmente.
    tam_disp = [300 + SG.degree(n) * 40 for n in disp_sub]
    tam_cta = [150 + SG.degree(n) * 30 for n in cta_sub]

    nx.draw_networkx_nodes(SG, pos, nodelist=disp_sub, node_color="#E74C3C",
                            node_size=tam_disp, label="Dispositivo sospechoso", alpha=0.9)
    nx.draw_networkx_nodes(SG, pos, nodelist=cta_sub, node_color="#3498DB",
                            node_size=tam_cta, label="Cuenta", alpha=0.7)

    # El grosor de cada arista refleja la frecuencia de transacciones
    # entre ese dispositivo y esa cuenta.
    anchos = [SG[u][v]["frecuencia"] * 0.8 for u, v in SG.edges()]
    nx.draw_networkx_edges(SG, pos, width=anchos, edge_color="gray", alpha=0.5)

    # Las etiquetas de texto solo se muestran para dispositivos (no para
    # cuentas, para no saturar visualmente el gráfico), quitando el
    # prefijo "DISP::" para que se lea más limpio.
    etiquetas_disp = {n: n.replace("DISP::", "") for n in disp_sub}
    nx.draw_networkx_labels(SG, pos, labels=etiquetas_disp, font_size=8, font_weight="bold")

    plt.title(f"Dispositivos sospechosos (grado >= {umbral}) y sus cuentas",
              fontsize=14, fontweight="bold")
    plt.legend(scatterpoints=1, loc="upper right")
    plt.axis("off")
    plt.tight_layout()

    if guardar_como:
        plt.savefig(guardar_como, dpi=150, bbox_inches="tight")
        print(f"\nGrafo visual guardado en: {guardar_como}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# HEATMAP: Dispositivo x Ventana de tiempo (densidad de actividad)
# ---------------------------------------------------------------------------
def dibujar_heatmap_actividad(df: pd.DataFrame, top_n: int = 15,
                               n_bins: int = 20, guardar_como: str = None):
    """
    Heatmap: filas = dispositivos con más transacciones, columnas = ventanas
    de tiempo (bins de TransactionDT), color = cantidad de transacciones.
    Revela ráfagas de actividad concentradas en el tiempo (patrón de bot/fraude).

    Args:
        df (pd.DataFrame): dataframe con columnas "Dispositivo" y "TransactionDT".
        top_n (int): cantidad de dispositivos más activos a graficar.
        n_bins (int): número de ventanas de tiempo en que se divide el rango.
        guardar_como (str, opcional): ruta de salida del PNG.
    """
    # Top N dispositivos con más transacciones (para que el heatmap sea legible).
    top_dispositivos = df["Dispositivo"].value_counts().head(top_n).index
    df_top = df[df["Dispositivo"].isin(top_dispositivos)].copy()

    # pd.cut discretiza el rango continuo de TransactionDT en n_bins
    # intervalos de igual ancho, etiquetados como 0, 1, 2... n_bins-1.
    df_top["bin_tiempo"] = pd.cut(df_top["TransactionDT"], bins=n_bins, labels=False)

    # Tabla pivote: dispositivo x bin_tiempo -> cantidad de transacciones
    # (aggfunc="count" cuenta cuántas filas caen en cada combinación).
    tabla = df_top.pivot_table(
        index="Dispositivo", columns="bin_tiempo",
        values="TransactionID", aggfunc="count", fill_value=0
    )

    plt.figure(figsize=(14, 7))
    sns.heatmap(tabla, cmap="Reds", linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Cantidad de transacciones"})
    plt.title(f"Heatmap de actividad: Top {top_n} dispositivos x ventana de tiempo",
              fontsize=13, fontweight="bold")
    plt.xlabel("Ventana de tiempo (bin de TransactionDT)")
    plt.ylabel("Dispositivo")
    plt.tight_layout()

    if guardar_como:
        plt.savefig(guardar_como, dpi=150, bbox_inches="tight")
        print(f"\nHeatmap guardado en: {guardar_como}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# HEATMAP INTERACTIVO (Plotly) - TODOS los dispositivos, con zoom/scroll
# ---------------------------------------------------------------------------
def dibujar_heatmap_interactivo(df: pd.DataFrame, n_bins: int = 20,
                                 guardar_como: str = "heatmap_interactivo.html"):
    """
    Heatmap interactivo con TODOS los dispositivos (sin recortar a top_n).
    Se abre en el navegador con zoom, scroll y tooltips al pasar el mouse.
    Útil cuando se quiere explorar el dataset completo, no solo un resumen.

    Args:
        df (pd.DataFrame): dataframe con columnas "Dispositivo" y "TransactionDT".
        n_bins (int): número de ventanas de tiempo en que se divide el rango.
        guardar_como (str): ruta del archivo HTML de salida.
    """
    df_copia = df.copy()
    df_copia["bin_tiempo"] = pd.cut(df_copia["TransactionDT"], bins=n_bins, labels=False)

    tabla = df_copia.pivot_table(
        index="Dispositivo", columns="bin_tiempo",
        values="TransactionID", aggfunc="count", fill_value=0
    )

    # px.imshow renderiza una matriz como mapa de calor interactivo.
    fig = px.imshow(
        tabla,
        labels=dict(x="Ventana de tiempo", y="Dispositivo", color="Transacciones"),
        color_continuous_scale="Reds",
        aspect="auto",
        title=f"Heatmap interactivo: {tabla.shape[0]} dispositivos x {n_bins} ventanas de tiempo"
    )
    # La altura de la figura escala con la cantidad de dispositivos,
    # para que cada fila tenga un alto mínimo legible.
    fig.update_layout(height=max(400, tabla.shape[0] * 18))

    fig.write_html(guardar_como)
    print(f"\nHeatmap interactivo guardado en: {guardar_como} (ábrelo en tu navegador)")


# ---------------------------------------------------------------------------
# RECORRIDO: BFS PARA COMPONENTES CONEXAS
# ---------------------------------------------------------------------------
def bfs_componente(G: nx.Graph, nodo_inicio: str) -> set:
    """BFS manual (no usamos nx.bfs_tree directamente, para mostrar la lógica).

    Recorre el grafo en anchura partiendo de nodo_inicio y devuelve
    el conjunto de todos los nodos alcanzables (es decir, toda la
    componente conexa a la que pertenece nodo_inicio).

    Args:
        G (nx.Graph): grafo sobre el que se recorre.
        nodo_inicio (str): nodo desde el cual arranca el BFS.

    Returns:
        set: conjunto de nodos visitados (la componente conexa completa).
    """
    visitados = {nodo_inicio}
    # deque permite popleft() en O(1); con una lista normal,
    # pop(0) sería O(n) en cada paso.
    cola = deque([nodo_inicio])

    while cola:
        actual = cola.popleft()
        for vecino in G.neighbors(actual):
            if vecino not in visitados:
                visitados.add(vecino)
                cola.append(vecino)

    return visitados


def encontrar_componentes(G: nx.Graph) -> list:
    """Encuentra todas las componentes conexas del grafo usando BFS.

    Recorre todos los nodos del grafo; si un nodo no ha sido visitado
    todavía, lanza un BFS desde ahí para descubrir toda su componente,
    y la agrega a la lista de resultados.

    Args:
        G (nx.Graph): grafo completo Dispositivo-Cuenta.

    Returns:
        list[set]: lista de componentes conexas (cada una es un set de nodos).
    """
    visitados_global = set()
    componentes = []

    for nodo in G.nodes():
        if nodo not in visitados_global:
            comp = bfs_componente(G, nodo)
            # |= es la unión de sets in-place (equivalente a .update()).
            visitados_global |= comp
            componentes.append(comp)

    return componentes


# ---------------------------------------------------------------------------
# CONCLUSIONES / DETECCIÓN
# ---------------------------------------------------------------------------
def generar_conclusiones(G: nx.Graph, componentes: list):
    """Imprime en consola un resumen interpretativo de los resultados del escenario.

    Args:
        G (nx.Graph): grafo Dispositivo-Cuenta ya construido.
        componentes (list[set]): componentes conexas halladas por encontrar_componentes().
    """
    print("\n" + "=" * 70)
    print("CONCLUSIONES - Escenario 1: Dispositivo -> Cuenta")
    print("=" * 70)

    print(f"\nTotal de nodos: {G.number_of_nodes()}")
    print(f"Total de aristas: {G.number_of_edges()}")
    print(f"Total de componentes conexas: {len(componentes)}")

    # Dispositivos ordenados por grado (cantidad de cuentas distintas).
    dispositivos = [n for n, d in G.nodes(data=True) if d["tipo"] == "Dispositivo"]
    # key=lambda n: G.degree(n) -> ordena de mayor a menor grado.
    ranking = sorted(dispositivos, key=lambda n: G.degree(n), reverse=True)

    print(f"\nTop 5 dispositivos con más cuentas asociadas:")
    for disp in ranking[:5]:
        grado = G.degree(disp)
        marca = "SOSPECHOSO" if grado >= UMBRAL_CUENTAS_SOSPECHOSAS else ""
        print(f"   {disp}: {grado} cuenta(s){marca}")

    # Componentes anómalamente grandes (más de 4 nodos en total,
    # contando tanto dispositivos como cuentas).
    componentes_grandes = [c for c in componentes if len(c) > 4]
    print(f"\nComponentes con más de 4 nodos (posibles fraud rings): {len(componentes_grandes)}")
    for i, comp in enumerate(componentes_grandes[:5], 1):
        n_disp = sum(1 for n in comp if n.startswith("DISP::"))
        n_cta = sum(1 for n in comp if n.startswith("CTA::"))
        print(f"   Componente {i}: {n_disp} dispositivo(s), {n_cta} cuenta(s)")

    print("\nInterpretación:")
    print("  - Dispositivos marcados como SOSPECHOSO controlan un número")
    print(f"    de cuentas (>= {UMBRAL_CUENTAS_SOSPECHOSAS}) muy por encima del comportamiento")
    print("    típico de un usuario legítimo (1-3 cuentas).")
    print("  - Componentes grandes sugieren posibles anillos de fraude:")
    print("    múltiples dispositivos y cuentas conectados transitivamente.")


# ---------------------------------------------------------------------------
# MENÚ PRINCIPAL
# ---------------------------------------------------------------------------
def ejecutar_escenario_1():
    """Orquesta el flujo completo del Escenario 1: carga, construcción del
    grafo, recorrido BFS, conclusiones y generación de visualizaciones."""
    print("\nCargando dataset...")
    df = cargar_datos(RUTA_TRANSACTION, RUTA_IDENTITY)
    print(f"Dataset cargado: {len(df)} registros.")

    print("\nConstruyendo grafo Dispositivo -> Cuenta...")
    G = construir_grafo(df)

    dibujar_grafo_ascii(G)

    print("\nEjecutando BFS para encontrar componentes conexas...")
    componentes = encontrar_componentes(G)

    generar_conclusiones(G, componentes)

    print("\nGenerando visualización gráfica de dispositivos sospechosos...")
    dibujar_grafo_visual(G, UMBRAL_CUENTAS_SOSPECHOSAS, guardar_como="grafo_escenario1.png")

    print("\nGenerando heatmap de actividad por dispositivo y tiempo...")
    dibujar_heatmap_actividad(df, guardar_como="heatmap_escenario1.png")

    print("\nGenerando heatmap interactivo (todos los dispositivos)...")
    dibujar_heatmap_interactivo(df, guardar_como="heatmap_interactivo.html")