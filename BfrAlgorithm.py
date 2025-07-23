#Carga de librerías necesarias
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from kneebow.rotor import Rotor
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


def cargar_datos_por_bloques(path, tamaño_bloque=100):
    for chunk in pd.read_csv(path, chunksize=tamaño_bloque, usecols=[0,1]):
        yield chunk.to_numpy()

class BFR:
    def __init__(self, K, umbral_distancia=2.0, d=2.0):
        self.K = K # K es el número de clústers en los que se desea particionar los datos
        self.d = d # Dimensión del dataset
        self.umbral_distancia = umbral_distancia # Umbral sobre distancia de Mahalanobis para determinar
                                                 # cuándo un punto se asignará a DS

        # DS: resumen por cluster (N, SUM, SUMSQ)
        self.DS = {}  # {cluster_id: {'N': ..., 'SUM': ..., 'SUMSQ': ...}}

        # CS: subclusters temporales
        self.CS = {}  # {cs_id: {'N': ..., 'SUM': ..., 'SUMSQ': ...}}

        # RS: puntos sin agrupar
        self.RS = []  # lista de puntos

        self.cs_id_counter = 0

    def dist_mahalanobis(self, punto, N, SUM, SUMSQ):
        centroide = SUM / N
        varianza = (SUMSQ / N) - (centroide ** 2)
        std_dev = np.sqrt(np.maximum(varianza, 1e-10))
        return np.sqrt(np.sum(((punto - centroide) / std_dev) ** 2))

    def actualizar_resumen(self, resumen, puntos):
      puntos = np.array(puntos)  # <-- conversión necesaria
      N = len(puntos)
      SUM = np.sum(puntos, axis=0)
      SUMSQ = np.sum(puntos ** 2, axis=0)
      resumen['N'] += N
      resumen['SUM'] += SUM
      resumen['SUMSQ'] += SUMSQ

    def inicializar(self, primer_bloque):
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(primer_bloque)
        labels = kmeans.labels_
        self.d = primer_bloque.shape[1]
        self.umbral_distancia = 2 * np.sqrt(self.d)

        for k in range(self.K):
            puntos_k = primer_bloque[labels == k]
            if len(puntos_k) <= 1:
                self.RS.extend(puntos_k)
            else:
                N = len(puntos_k)
                SUM = np.sum(puntos_k, axis=0)
                SUMSQ = np.sum(puntos_k ** 2, axis=0)
                self.DS[k] = {'N': N, 'SUM': SUM, 'SUMSQ': SUMSQ}

        # Clusterizamos los RS iniciales
        self._procesar_RS()

    def _procesar_RS(self):
        if len(self.RS) < 2:
            return
        try:
            n_clusters = max(1, len(self.RS) // 10)
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.RS)
        except Exception:
            return

        nuevos_RS = []
        for i in range(n_clusters):
            puntos = np.array(self.RS)[kmeans.labels_ == i]
            if len(puntos) <= 1:
                nuevos_RS.extend(puntos)
            else:
                resumen = {
                    'N': len(puntos),
                    'SUM': np.sum(puntos, axis=0),
                    'SUMSQ': np.sum(puntos ** 2, axis=0)
                }
                self.CS[self.cs_id_counter] = resumen
                self.cs_id_counter += 1
        self.RS = nuevos_RS
        if len(self.CS) > 1:
          self._fusionar_CS()

    def _asignar_a_clusters(self, bloque):
        nuevos_RS = []
        for punto in bloque:
            asignado = False
            for k, resumen in self.DS.items():
                dist = self.dist_mahalanobis(punto, **resumen)
                if dist < self.umbral_distancia:
                    self.actualizar_resumen(resumen, [punto])
                    asignado = True
                    break
            if not asignado:
                nuevos_RS.append(punto)
        self.RS.extend(nuevos_RS)
        self._procesar_RS()
        self._fusionar_CS()

    def _fusionar_CS(self, umbral_varianza_total: float = 1.0):
        keys = list(self.CS.keys())
        n = len(keys)
        eliminados = set()

        for i in range(n):
            if keys[i] in eliminados:
                continue
            for j in range(i + 1, n):
                if keys[j] in eliminados:
                    continue

                cs1 = self.CS[keys[i]]
                cs2 = self.CS[keys[j]]

                # Fusionamos estadísticas
                N_total = cs1["N"] + cs2["N"]
                SUM_total = cs1["SUM"] + cs2["SUM"]
                SUMSQ_total = cs1["SUMSQ"] + cs2["SUMSQ"]

                # Calculamos varianza combinada por dimensión
                media = SUM_total / N_total
                varianza = (SUMSQ_total / N_total) - (media ** 2)

                # Evaluamos si la suma de varianzas es aceptable
                varianza_total = np.sum(varianza)

                if varianza_total < umbral_varianza_total:
                    # Aquí se fusionan los CS
                    self.CS[keys[i]] = {
                        "N": N_total,
                        "SUM": SUM_total,
                        "SUMSQ": SUMSQ_total
                    }
                    eliminados.add(keys[j])

        # Eliminamos los CS fusionados
        for clave in eliminados:
            del self.CS[clave]

    def procesar_bloques(self, path_archivo):
        bloques = cargar_datos_por_bloques(path_archivo)

        # Inicializamos con el primer bloque
        bloque_inicial = next(bloques)
        self.inicializar(bloque_inicial)

        for bloque in bloques:
            self._asignar_a_clusters(bloque)

        self.finalizar()


    def finalizar(self):
      # 1. Movemos cada CS al DS más cercano
      for cs_id, resumen_cs in list(self.CS.items()):
          centroide_cs = resumen_cs['SUM'] / resumen_cs['N']
          mejor_dist = float('inf')
          mejor_ds_id = None

          for ds_id, resumen_ds in self.DS.items():
              dist = self.dist_mahalanobis(centroide_cs, **resumen_ds)
              if dist < mejor_dist:
                  mejor_dist = dist
                  mejor_ds_id = ds_id

          if mejor_ds_id is not None:
              self.DS[mejor_ds_id]['N'] += resumen_cs['N']
              self.DS[mejor_ds_id]['SUM'] += resumen_cs['SUM']
              self.DS[mejor_ds_id]['SUMSQ'] += resumen_cs['SUMSQ']
                # Actualizamos con N copias del centroide
          del self.CS[cs_id]  # Quita del CS

      # 2. Intentamos asignar cada punto del RS a algún DS
      nuevos_outliers = []
      for punto in self.RS:
          asignado = False
          for ds_id, resumen in self.DS.items():
              dist = self.dist_mahalanobis(punto, **resumen)
              if dist < self.umbral_distancia:
                  self.actualizar_resumen(resumen, [punto])
                  asignado = True
                  break
          if not asignado:
              nuevos_outliers.append(punto)
      self.RS = nuevos_outliers

      # 3. Impresión
      print(f"Clusters finales en DS: {len(self.DS)}")
      print(f"Outliers definitivos en RS: {len(self.RS)}")


file_path = "file_2.csv" # Nombre del archivo
k = 3 # K clústers
bfr = BFR(K=k)
bfr.procesar_bloques(file_path)



fig, ax = plt.subplots(figsize=(8, 6))

for cluster_id, resumen in bfr.DS.items():
    N = resumen['N']
    SUM = resumen['SUM']
    SUMSQ = resumen['SUMSQ']

    if N == 0:
        continue

    # Centroide
    mu = SUM / N

    # Varianzas
    var = (SUMSQ / N) - (mu ** 2)
    std = np.sqrt(var)

    # Dibuja el elipse (en 2D)
    ellipse = Ellipse(
        xy=mu[:2],  # solo si estás en 2D
        width=2*std[0],  # 2 desviaciones estándar en x
        height=2*std[1],  # 2 desviaciones estándar en y
        edgecolor='black',
        facecolor='none'
    )
    ax.add_patch(ellipse)
    ax.plot(mu[0], mu[1], 'o', label=f'Cluster {cluster_id}')

ax.set_title('Centroide y dispersión de clústeres (DS)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.grid(True)
ax.legend()
plt.show()


# Cargar todos los datos originales
df = pd.read_csv(file_path, usecols=[0, 1])
puntos = df.to_numpy()

def calcular_mahalanobis(punto, resumen):
    N = resumen['N']
    SUM = resumen['SUM']
    SUMSQ = resumen['SUMSQ']

    mu = SUM / N
    var = (SUMSQ / N) - (mu ** 2)
    var[var == 0] = 1e-10  # evitar división por cero
    dist = np.sqrt(np.sum(((punto - mu) ** 2) / var))
    return dist
# Asignar cada punto a su cluster más cercano en DS
labels = []

for punto in puntos:
    mejor_cluster = None
    mejor_distancia = float('inf')
    for cluster_id, resumen in bfr.DS.items():
        dist = calcular_mahalanobis(punto, resumen)
        if dist < mejor_distancia:
            mejor_distancia = dist
            mejor_cluster = cluster_id

    # Si no cumple con el umbral, lo etiquetamos como "outlier" (-1)
    if mejor_distancia < bfr.umbral_distancia:
        labels.append(mejor_cluster)
    else:
        labels.append(-1)  # outlier


# Convertimos todo a arrays numpy
puntos = np.array(puntos)
labels = np.array(labels)

# Graficar
plt.figure(figsize=(8, 6))
clusters_unicos = np.unique(labels)

for cid in clusters_unicos:
    if cid == -1:
        plt.scatter(puntos[labels == cid, 0], puntos[labels == cid, 1],
                    color='black', label='Outliers')
    else:
        plt.scatter(puntos[labels == cid, 0], puntos[labels == cid, 1],
                    label=f'Cluster {cid}')


for cid, resumen in bfr.DS.items():
    mu = resumen['SUM'] / resumen['N']
    plt.scatter(mu[0], mu[1], color='black', marker='X', s=200, edgecolor='black', label=f'Centroide {cid}')

plt.title('Asignación final de puntos al DS con Mahalanobis (BFR)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
kmeans.fit(df)
predictions = kmeans.labels_

plt.figure(figsize=(10, 6))
scatter_plot = sns.scatterplot(
    data=df, x='x', y='y',
    hue=predictions,
    palette='pastel',
    s=100,
    edgecolor='k',
    alpha=0.7
)

# Centroides
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0], centroids[:, 1],
    s=300,
    c='black',
    marker='X',
    edgecolor='black',
    linewidth=2,
    label='Centroides'
)

plt.title('Resultado de KMeans', fontsize=16)
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.legend(title='Cluster', loc='upper right')

plt.show()

def calculate_kn_distance(X, neigh = 2):
	neigh = NearestNeighbors(n_neighbors = neigh)
	nbrs = neigh.fit(X)
	distances, indices = nbrs.kneighbors(X)
	return distances[:, 1:].reshape(-1)

def get_eps(X, neigh=2):
	eps_dist = np.sort(calculate_kn_distance(X, neigh=neigh))
	rotor = Rotor()
	curve_xy = np.concatenate([np.arange(eps_dist.shape[0]).reshape(-1, 1), eps_dist.reshape(-1,1)],1)
	rotor.fit_rotate(curve_xy)
	rotor.plot_elbow()
	e_idx = rotor.get_elbow_index()
	return curve_xy[e_idx]
idx, eps = get_eps(df)
plt.show()
print(eps)

dbscan = DBSCAN(eps=0.7237068410065157, min_samples=3).fit(df) # min_samples es otra forma de llamar a minpoints
labelsDBS = dbscan.labels_

# Número de clústeres
n_clusters = len(set(labelsDBS)) - (1 if -1 in labelsDBS else 0)

# Usa una paleta predefinida como 'tab10', 'Set1', etc.
palette = plt.get_cmap('tab10')  # hasta 10 colores distintos bien contrastados

# Asigna colores a los clústeres
cluster_colors = np.array([
    palette(cluster % 10)[:3] if cluster != -1 else [0, 0, 0]
    for cluster in labelsDBS
])
# Generación de gráfica
plt.figure(figsize=(10, 5))
plt.scatter(df['x'], df['y'], c=cluster_colors, marker='o',
            linewidth=0.1, s=15)

plt.ticklabel_format(style='plain', useOffset=False, axis='y')
plt.title('Clusters DBSCAN')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.grid(True)
plt.show()


# Para K-Means
# predictions es un array con las etiquetas de clúster para cada punto
n_clusters = len(set(predictions)) - (1 if -1 in predictions else 0)
# Excluimos los outliers para BFR
mask_bfr = labels != -1
puntos_filtrados = puntos[mask_bfr]
labels_filtrados = labels[mask_bfr]

if n_clusters > 1:
    score = silhouette_score(df, predictions)
    print(f"Silhouette Score - KMeans (sin outliers): {score}")
else:
    print("No se puede calcular Silhouette Score: menos de dos clústeres reales.")

# Para DBSCAN (solo si hay más de 1 cluster)
if len(set(labelsDBS)) > 1 and -1 in labelsDBS:
    mask = labelsDBS != -1  # Excluye ruido
    score_dbscan = silhouette_score(df[mask], labelsDBS[mask])
    print("Silhouette Score - DBSCAN (sin outliers):", score_dbscan)
elif len(set(labelsDBS)) > 1:
    score_dbscan = silhouette_score(df, labelsDBS)
    print("Silhouette Score - DBSCAN:", score_dbscan)
else:
    print("DBSCAN encontró solo un cluster: Silhouette no se puede calcular.")

if len(np.unique(labels_filtrados)) > 1:
    score_bfr = silhouette_score(puntos_filtrados, labels_filtrados)
    print("Silhouette Score - BFR (sin outliers):", score_bfr)
else:
    print("BFR encontró solo un cluster: no se puede calcular Silhouette.")

