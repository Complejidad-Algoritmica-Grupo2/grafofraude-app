"""
Menu principal del sistema de deteccion de fraude GrafoFraude.

Permite seleccionar y ejecutar los tres escenarios de analisis.

NOTA DE REVISION: este archivo adapta el menu.py original de Winnie
para funcionar en grafofraude-app. El unico cambio funcional es
reemplazar los imports relativos (from .algorithm...) por imports
absolutos (from algorithm...), necesario porque menu.py vive en la
raiz del proyecto y no dentro de un subpaquete.

El resto del codigo (logica del menu, try/except, textos) es fiel
al original.
"""

# Imports absolutos (sin punto inicial).
# En el repo original de Winnie eran relativos: from .algorithm.scenario1 ...
# porque menu.py vivia dentro del paquete fraud_detector_program/.
# En grafofraude-app, menu.py esta en la raiz, asi que se importa directamente.
import os

from algorithm.scenario1 import ejecutar_escenario_1
from algorithm.scenario2 import ejecutar_escenario_2
from algorithm.scenario3 import ejecutar_escenario_3

# Carpeta donde los 3 escenarios guardan sus PNG/HTML generados.
# Se crea una sola vez al iniciar el menú.
# os.makedirs() dentro de cada scenarioN.py por separado.
CARPETA_OUTPUTS = "/home/GrafoFraude/grafofraude-app/outputs"

def menu():
    """Muestra el menu principal en consola y ejecuta el escenario elegido.

    Corre en un bucle infinito hasta que el usuario elige la opcion 0 (Salir).
    Cada escenario esta envuelto en try/except para que un error interno
    no tumbe el programa completo — el usuario puede seguir eligiendo opciones.
    """
    os.makedirs(CARPETA_OUTPUTS, exist_ok=True)

    while True:
        print("\n" + "=" * 70)
        print(" SISTEMA DE DETECCION DE FRAUDE - MENU PRINCIPAL")
        print("=" * 70)
        print(" 1) Escenario 1: Dispositivo -> Cuenta (BFS / Componentes conexas)")
        print(" 2) Escenario 2: Tarjeta -> Transacciones (Programacion dinamica)")
        print(" 3) Escenario 3: Identidad -> Cuenta (Algoritmo de Louvain)")
        print(" 0) Salir")

        opcion = input("\nElige una opcion: ").strip()

        if opcion == "1":
            try:
                # Ejecuta el escenario completo: carga datos, construye grafo,
                # corre BFS, imprime conclusiones y genera visualizaciones.
                ejecutar_escenario_1()
            except Exception as e:
                # Si algo falla (ej. CSV no encontrado, columna faltante),
                # se imprime el error y el menu vuelve a aparecer.
                print(f"\n Error en Escenario 1: {e}")
            input("\nPresiona ENTER para volver al menú principal...")

        elif opcion == "2":
            try:
                # Ejecuta el escenario completo: carga datos, calcula scores,
                # corre DP de Kadane por tarjeta, imprime conclusiones y grafica.
                ejecutar_escenario_2()
            except Exception as e:
                print(f"\n Error en Escenario 2: {e}")
            input("\nPresiona ENTER para volver al menú principal...")

        elif opcion == "3":
            try:
                # Ejecuta el escenario completo: carga datos, construye grafo,
                # corre Louvain, imprime conclusiones y genera visualizacion.
                ejecutar_escenario_3()
            except Exception as e:
                print(f"\n Error en Escenario 3: {e}")
            input("\nPresiona ENTER para volver al menú principal...")

        elif opcion == "0":
            print("Saliendo del sistema...")
            break

        else:
            # Cualquier entrada que no sea 0, 1, 2 o 3.
            print(" Opcion no disponible. Elige 1, 2, 3 o 0.")

# Sin este bloque, menu() no se ejecuta automaticamente al correr
# "python menu.py" — era el bug documentado en el repo original de Winnie.
# Con if __name__ == "__main__", el menu arranca solo al ejecutar
# el archivo directamente, pero no cuando se importa desde otro modulo.
if __name__ == "__main__":
    menu()
