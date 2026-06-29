"""
Sistema de Detección de Fraude basado en Grafos
Escenario 3: Identidad -> Cuenta
Algoritmo: Detección de comunidades (Louvain)

Idea central: una identidad (real o sintética) que controla muchas cuentas
genera un subgrafo denso. Louvain encuentra agrupaciones de nodos más
conectados entre sí que con el resto de la red, sin necesitar un nodo
semilla (a diferencia de BFS).

Dataset esperado (CSV): columnas mínimas requeridas
    Transaction: TransactionID, card1, card2, addr1
    Identity:    TransactionID, id_12 ... id_38 (se usan las disponibles)

NOTA DE REVISIÓN: este archivo conserva la lógica original de Winnie
sin modificaciones. Los comentarios fueron agregados únicamente con
fines de comprensión/documentación (sin alterar el comportamiento).
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
# louvain_communities está incluida en networkx >= 2.8 — NO requiere
# instalar python-louvain ni community por separado. Solo necesitas
# que networkx esté actualizado (ya está en requirements.txt).
from networkx.algorithms.community import louvain_communities

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# - Definir las variables globales
# ---------------------------------------------------------------------------
# Rutas relativas al CSV (desde donde se ejecuta el programa).
RUTA_TRANSACTION = "fraud_detector_program/data/train_transaction.csv"
RUTA_IDENTITY = "fraud_detector_program/data/train_identity.csv"

# Mismo tamaño de muestra que los otros dos escenarios para coherencia.
N_REGISTROS = 1500

# Umbral de sospecha a nivel de nodo "Identidad" (cuántas cuentas distintas
# controla antes de considerarse anómala).
UMBRAL_CUENTAS_SOSPECHOSAS = 5

# Umbral de sospecha a nivel de comunidad (cuántos nodos totales —
# identidades + cuentas — debe tener una comunidad para reportarse).
TAMANO_COMUNIDAD_SOSPECHOSA = 4


# ---------------------------------------------------------------------------
# CARGA Y PREPARACIÓN DE DATOS
# ---------------------------------------------------------------------------
def cargar_datos(ruta_transaction: str, ruta_identity: str,
                  n_muestra: int = N_REGISTROS) -> pd.DataFrame:
    """
    Carga transaction.csv e identity.csv, hace LEFT JOIN por TransactionID,
    excluye transacciones sin datos de identidad, y toma una muestra de
    n_muestra registros sobre ese subset.

    Args:
        ruta_transaction (str): ruta al CSV de transacciones.
        ruta_identity (str): ruta al CSV de identidad.
        n_muestra (int): cantidad máxima de registros a usar (default 1500).

    Returns:
        pd.DataFrame: dataframe muestreado con columnas proxy
            "Cuenta" e "Identidad" agregadas.

    Raises:
        ValueError: si el CSV de identity no contiene ninguna columna
            con prefijo "id_" o "id-".
    """
    # Solo las columnas mínimas de transacción que este escenario necesita.
    cols_transaction = ["TransactionID", "card1", "card2", "addr1"]
    df_transaction = pd.read_csv(ruta_transaction, usecols=cols_transaction)

    # Detección dinámica de columnas de identidad (id_12..id_38 o id-12..id-38).
    # El dataset de Kaggle puede venir con guion bajo (train) o guion (test),
    # así que se detectan ambas variantes para que el código funcione en los dos casos.
    columnas_disponibles = pd.read_csv(ruta_identity, nrows=0).columns.tolist()
    columnas_id = [c for c in columnas_disponibles
                   if c.startswith("id_") or c.startswith("id-")]

    if not columnas_id:
        raise ValueError(
            "No se encontraron columnas id_12..id_38 (ni id-12..id-38) "
            "en el archivo de identity."
        )

    # Se leen TODAS las columnas de identidad detectadas (no se sabe de antemano
    # cuáles tienen datos útiles vs. cuáles están casi vacías).
    cols_identity = ["TransactionID"] + columnas_id
    df_identity = pd.read_csv(ruta_identity, usecols=cols_identity)

    # LEFT JOIN: se conservan todas las transacciones aunque no tengan
    # fila en identity.csv (quedarán con NaN en las columnas id_XX).
    df = df_transaction.merge(df_identity, on="TransactionID", how="left")

    # Excluir transacciones donde TODAS las columnas de identidad son nulas.
    # how="all" → elimina la fila solo si TODAS son nulas simultáneamente;
    # si al menos una tiene valor, la fila se conserva.
    filas_antes = len(df)
    df = df.dropna(subset=columnas_id, how="all")
    filas_excluidas = filas_antes - len(df)
    print(f"Transacciones sin datos de identidad excluidas: {filas_excluidas} de {filas_antes}")

    # Muestra reproducible (random_state=42) tomada YA SOBRE el subset
    # que sí tiene al menos un dato de identidad.
    n = min(n_muestra, len(df))
    df = df.sample(n=n, random_state=42).reset_index(drop=True)
    print(f"Muestra tomada: {n} registros "
          f"(de {filas_antes - filas_excluidas} con identidad disponible)")

    # Rellenar nulos para que la concatenación de strings no genere "nan"
    # dentro de los identificadores proxy.
    df = df.fillna({"card1": "NA", "card2": "NA", "addr1": "NA"})
    for c in columnas_id:
        df[c] = df[c].fillna("NA")

    # Nodo proxy "Cuenta": mismo criterio que Escenario 1.
    df["Cuenta"] = (
        df["card1"].astype(str) + "_" +
        df["card2"].astype(str) + "_" +
        df["addr1"].astype(str)
    )

    # Nodo proxy "Identidad": concatena TODAS las columnas id_XX disponibles.
    # agg("_".join, axis=1) une los valores de cada fila con "_" como separador,
    # formando un string único que representa la "huella de identidad" de esa
    # transacción. Dos transacciones con la misma combinación de valores id_XX
    # se consideran de la misma identidad.
    df["Identidad"] = df[columnas_id].astype(str).agg("_".join, axis=1)

    return df


# ---------------------------------------------------------------------------
# CONSTRUCCIÓN DEL GRAFO
# ---------------------------------------------------------------------------
def construir_grafo(df: pd.DataFrame) -> nx.Graph:
    """Construye un grafo bipartito Identidad-Cuenta.

    Cada nodo de tipo "Identidad" representa una huella de identidad proxy.
    Cada nodo de tipo "Cuenta" representa una cuenta proxy (card1_card2_addr1).
    La arista entre ambos tiene atributo "frecuencia" = cuántas transacciones
    vincularon esa identidad a esa cuenta en la muestra.

    Args:
        df (pd.DataFrame): dataframe con columnas "Identidad" y "Cuenta".

    Returns:
        nx.Graph: grafo no dirigido bipartito con nodos tipados y
            aristas con atributo "frecuencia".
    """
    G = nx.Graph()

    for _, fila in df.iterrows():
        # Prefijos "ID::" y "CTA::" evitan colisiones de nombre si un
        # string pudiera representar tanto una identidad como una cuenta.
        ident = f"ID::{fila['Identidad']}"
        cuenta = f"CTA::{fila['Cuenta']}"

        # add_node es idempotente — si el nodo ya existe, no lo duplica.
        G.add_node(ident, tipo="Identidad")
        G.add_node(cuenta, tipo="Cuenta")

        if G.has_edge(ident, cuenta):
            # Acumular frecuencia si ya existe la arista.
            G[ident][cuenta]["frecuencia"] += 1
        else:
            G.add_edge(ident, cuenta, frecuencia=1)

    return G


# ---------------------------------------------------------------------------
# DIBUJO EN CONSOLA (ASCII)
# ---------------------------------------------------------------------------
def dibujar_grafo_ascii(G: nx.Graph, max_identidades: int = 10):
    """Imprime una lista de adyacencia en consola, limitada para no saturarla.

    Args:
        G (nx.Graph): grafo Identidad-Cuenta ya construido.
        max_identidades (int): cantidad máxima de identidades a mostrar.
    """
    print("\n" + "=" * 70)
    print("GRAFO (vista ASCII) - Identidad -> Cuenta")
    print("=" * 70)

    # Filtrar solo nodos de tipo Identidad.
    identidades = [n for n, d in G.nodes(data=True) if d["tipo"] == "Identidad"]

    for ident in identidades[:max_identidades]:
        vecinos = list(G.neighbors(ident))
        print(f"\n[{ident}]  (grado={len(vecinos)})")
        for cuenta in vecinos[:10]:
            freq = G[ident][cuenta]["frecuencia"]
            print(f"   └──> {cuenta}   (frecuencia={freq})")
        if len(vecinos) > 10:
            print(f"   ... y {len(vecinos) - 10} cuentas más")

    if len(identidades) > max_identidades:
        print(f"\n... y {len(identidades) - max_identidades} identidades más (omitidas)")


# ---------------------------------------------------------------------------
# RECORRIDO: DETECCIÓN DE COMUNIDADES (LOUVAIN)
# ---------------------------------------------------------------------------
def detectar_comunidades(G: nx.Graph) -> list:
    """
    Aplica el algoritmo de Louvain sobre el grafo Identidad-Cuenta y
    devuelve las comunidades encontradas.

    Louvain optimiza la modularidad del grafo: agrupa nodos que tienen
    más conexiones ENTRE SÍ que con el resto de la red.
    A diferencia de BFS (Escenario 1), NO necesita un nodo semilla —
    analiza la red completa de una sola vez.

    Args:
        G (nx.Graph): grafo Identidad-Cuenta completo.

    Returns:
        list[set]: lista de comunidades, cada una es un set de nodos.
            El orden no es significativo (Louvain es no determinista;
            seed=42 lo hace reproducible entre ejecuciones).
    """
    # weight="frecuencia" indica que Louvain debe considerar el peso de
    # cada arista (cuántas transacciones vincularon esa identidad-cuenta)
    # al calcular la modularidad: pares con más transacciones tienen más
    # "afinidad" y es más probable que queden en la misma comunidad.
    # seed=42 fija la aleatoriedad interna del algoritmo para reproducibilidad.
    comunidades = louvain_communities(G, weight="frecuencia", seed=42)
    return comunidades


# ---------------------------------------------------------------------------
# VISUALIZACIÓN: comunidades coloreadas
# ---------------------------------------------------------------------------
def dibujar_comunidades(G: nx.Graph, comunidades: list, guardar_como: str = None):
    """
    Dibuja SOLO las comunidades de tamaño > TAMANO_COMUNIDAD_SOSPECHOSA,
    cada una con un color distinto, para resaltar agrupaciones anómalas.

    Args:
        G (nx.Graph): grafo Identidad-Cuenta completo.
        comunidades (list[set]): resultado de detectar_comunidades().
        guardar_como (str, opcional): si se indica, guarda el PNG en esa
            ruta en vez de abrir ventana interactiva con plt.show().
    """
    # Filtrar solo las comunidades que superan el umbral de tamaño.
    comunidades_grandes = [c for c in comunidades if len(c) > TAMANO_COMUNIDAD_SOSPECHOSA]

    if not comunidades_grandes:
        print(f"\nNo hay comunidades con más de {TAMANO_COMUNIDAD_SOSPECHOSA} nodos para graficar.")
        return

    # Reunir todos los nodos de las comunidades grandes para el subgrafo.
    # |= es unión de sets in-place (equivalente a .update()).
    nodos_a_graficar = set()
    for c in comunidades_grandes:
        nodos_a_graficar |= c

    # G.subgraph() devuelve una vista (no copia) restringida a esos nodos.
    SG = G.subgraph(nodos_a_graficar)

    # spring_layout distribuye los nodos en el espacio 2D intentando que
    # los más conectados queden más cerca (k=0.6 controla la distancia óptima).
    # seed=42 hace el layout reproducible.
    pos = nx.spring_layout(SG, seed=42, k=0.6)

    plt.figure(figsize=(13, 9))
    # tab10 tiene 10 colores distintos; el módulo (%) recicla si hay más
    # de 10 comunidades grandes.
    colores = plt.cm.tab10.colors

    for i, comunidad in enumerate(comunidades_grandes):
        color = colores[i % len(colores)]
        nx.draw_networkx_nodes(SG, pos, nodelist=list(comunidad), node_color=[color],
                                node_size=250, alpha=0.85,
                                label=f"Comunidad {i+1} ({len(comunidad)} nodos)")

    # Las aristas se dibujan al final, sin colores especiales (gris transparente),
    # para que los nodos coloreados sean el foco visual.
    nx.draw_networkx_edges(SG, pos, alpha=0.3, edge_color="gray")

    plt.title("Comunidades detectadas (Louvain) - Identidad x Cuenta",
              fontsize=13, fontweight="bold")
    plt.legend(scatterpoints=1, loc="upper right", fontsize=8)
    plt.axis("off")
    plt.tight_layout()

    if guardar_como:
        plt.savefig(guardar_como, dpi=150, bbox_inches="tight")
        print(f"\nGráfico de comunidades guardado en: {guardar_como}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CONCLUSIONES
# ---------------------------------------------------------------------------
def generar_conclusiones(G: nx.Graph, comunidades: list):
    """Imprime en consola un resumen interpretativo de los resultados del escenario.

    Args:
        G (nx.Graph): grafo Identidad-Cuenta ya construido.
        comunidades (list[set]): resultado de detectar_comunidades().
    """
    print("\n" + "=" * 70)
    print("CONCLUSIONES - Escenario 3: Identidad -> Cuenta")
    print("=" * 70)

    print(f"\nTotal de nodos: {G.number_of_nodes()}")
    print(f"Total de aristas: {G.number_of_edges()}")
    print(f"Total de comunidades detectadas: {len(comunidades)}")

    # Identidades ordenadas por grado (cuántas cuentas distintas controla cada una).
    identidades = [n for n, d in G.nodes(data=True) if d["tipo"] == "Identidad"]
    ranking = sorted(identidades, key=lambda n: G.degree(n), reverse=True)

    print(f"\nTop 5 identidades con más cuentas asociadas:")
    for ident in ranking[:5]:
        grado = G.degree(ident)
        marca = " SOSPECHOSO" if grado >= UMBRAL_CUENTAS_SOSPECHOSAS else ""
        print(f"   {ident}: {grado} cuenta(s){marca}")

    # Comunidades que superan el umbral de tamaño mínimo sospechoso.
    comunidades_grandes = [c for c in comunidades if len(c) > TAMANO_COMUNIDAD_SOSPECHOSA]
    print(f"\nComunidades con más de {TAMANO_COMUNIDAD_SOSPECHOSA} nodos: {len(comunidades_grandes)}")

    # Mostrar las 5 más grandes, ordenadas de mayor a menor tamaño.
    for i, c in enumerate(sorted(comunidades_grandes, key=len, reverse=True)[:5], 1):
        n_id = sum(1 for n in c if n.startswith("ID::"))
        n_cta = sum(1 for n in c if n.startswith("CTA::"))
        print(f"   Comunidad {i}: {n_id} identidad(es), {n_cta} cuenta(s)")

    print("\nInterpretación:")
    print("  - Identidades marcadas como SOSPECHOSO controlan un número de")
    print(f"    cuentas (>= {UMBRAL_CUENTAS_SOSPECHOSAS}) muy por encima de un usuario legítimo.")
    print("  - Comunidades grandes y densas sugieren posible identidad")
    print("    sintética o un grupo coordinado abriendo múltiples cuentas.")
    print("  - A diferencia de BFS (Escenario 1), Louvain no necesita un nodo")
    print("    semilla: encuentra agrupaciones densas en toda la red a la vez.")


# ---------------------------------------------------------------------------
# FLUJO PRINCIPAL DEL ESCENARIO
# ---------------------------------------------------------------------------
def ejecutar_escenario_3():
    """Orquesta el flujo completo del Escenario 3: carga, construcción del
    grafo, detección de comunidades Louvain, conclusiones y visualización."""
    print("\nCargando dataset...")
    df = cargar_datos(RUTA_TRANSACTION, RUTA_IDENTITY)
    print(f"Dataset cargado: {len(df)} registros.")

    print("\nConstruyendo grafo Identidad -> Cuenta...")
    G = construir_grafo(df)

    dibujar_grafo_ascii(G)

    print("\nEjecutando Louvain para detección de comunidades...")
    comunidades = detectar_comunidades(G)

    generar_conclusiones(G, comunidades)

    print("\nGenerando visualización de comunidades...")
    dibujar_comunidades(G, comunidades, guardar_como="comunidades_escenario3.png")


# Permite ejecutar este escenario directamente con:
# python algorithm/scenario3.py
if __name__ == "__main__":
    ejecutar_escenario_3()