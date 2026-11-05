import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import rgb2lab, lab2rgb, xyz2lab, lab2xyz
from PIL import Image, ImageEnhance
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import colour
from colour.colorimetry import sd_to_XYZ
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from scipy.interpolate import interp1d
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from itertools import product
from skimage.filters import threshold_otsu
# from cuml.cluster import HDBSCAN

def get_colour_matching_functions_and_illuminant():
    illuminant_spd = colour.SDS_ILLUMINANTS['D65']  # gets the spectral power distribuition in CIE (a standart for simulating daylight)
    # this is a standard illuminant defined by the CIE (International Commission on Illumination) to represent average daylight conditions. It is widely used in color science and computer graphics for color rendering and color matching.
    # intuitively, it represents the spectral distribution of light that we would expect to see on a clear day with the sun at a specific angle. It is important because different colors can appear differently under different lighting conditions, and D65 provides a standardized way to simulate daylight illumination.
    cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']   # the colour matching functions to each wavelength

    # Align the shape of illuminant and CMFs with intrapolations and extrapolations
    # CMFs wavelengths is (360nm, to 830nm, 1nm step), and illuminant is (300nm, to 780nm, 5nm step)
    illuminant_spd = illuminant_spd.copy().align(cmfs.shape)
    wavelengths = illuminant_spd.wavelengths
    illuminant_spd_values = illuminant_spd.values

    # gets the R, G, B matching functions for each wavelength
    x_bar = cmfs.values[:, 0]
    y_bar = cmfs.values[:, 1]
    z_bar = cmfs.values[:, 2]
    # stacks them into a single array in the shape (wavelengths, 3)
    xyz = np.stack([x_bar, y_bar, z_bar], axis=-1)

    plot_flag = True
    # segment to plot the illuminant and the CMFs
    if plot_flag:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(wavelengths, illuminant_spd_values, label='D65 SPD')
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Relative Power')
        ax1.set_title(f"{'D65'}  Illuminant Spectral Power Distribution")
        ax1.legend()

        ax2.plot(wavelengths, x_bar, label='x̄(λ)', color='r')
        ax2.plot(wavelengths, y_bar, label='ȳ(λ)', color='g')
        ax2.plot(wavelengths, z_bar, label='z̄(λ)', color='b')
        ax2.set_xlabel('Wavelength (nm)')   
        ax2.set_ylabel('Amplitude')
        ax2.set_title('CIE 1931 2° Standard Observer Color Matching Functions')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    return wavelengths, illuminant_spd_values, xyz

def generate_true_color_image(X, frequencies, spectralon_coords):

    height, width, bands = X.shape[1], X.shape[2], X.shape[0]
    Xnew = np.transpose(X, axes=(1, 2, 0)) 

    # Get standard illuminant and color matching functions from the function above
    std_wavelengths, illuminant_vals, cmfs_xyz = get_colour_matching_functions_and_illuminant()

    # Trim spectral bands to ~395–830 nm
    wl_min, wl_max = 395.0, 830.0
    band_centers = np.array(frequencies)
    band_mask = (band_centers >= wl_min) & (band_centers <= wl_max)
    band_centers_trimmed = band_centers[band_mask]
    X_trimmed = Xnew[:, :, band_mask] / 255.0  # Normalize reflectance

    # Interpolate illuminant and CMFs
    # the interpolation functions take the standard wavelengths and return the corresponding values at the trimmed band centers
    # does it actually interpolate? yes, because the band centers may not align exactly with the standard wavelengths
    illum_interp = interp1d(std_wavelengths, illuminant_vals, kind='linear', fill_value="extrapolate")(
        band_centers_trimmed)
    x = interp1d(std_wavelengths, cmfs_xyz[:, 0], kind='linear', fill_value='extrapolate')(band_centers_trimmed)
    y = interp1d(std_wavelengths, cmfs_xyz[:, 1], kind='linear', fill_value='extrapolate')(band_centers_trimmed)
    z = interp1d(std_wavelengths, cmfs_xyz[:, 2], kind='linear', fill_value='extrapolate')(band_centers_trimmed)
    xyz_interp = np.column_stack((x, y, z))

    # Flatten the image for matrix multiplication
    flat = X_trimmed.reshape(-1, X_trimmed.shape[2]).T  # (bands, H*W)
    # the first operation does the matrix multiplication between the interpolated CMFs and the diagonal matrix of the illuminant with the flattened reflectance data
    # which gives us the XYZ tristimulus values for each pixel (tristimulus values are a way to represent colors based on human vision)
    # it is the result of multiplying the color matching functions by the illuminant and the reflectance data
    # the color matching functions are in xyz_interp, the illuminant is in illum_interp, and the reflectance data is in flat
    XYZ = xyz_interp.T @ np.diag(illum_interp) @ flat  # (3, H*W)
    XYZ_image = XYZ.T.reshape(height, width, 3)
    # XYZ_image = (XYZ_image - XYZ_image.min()) / (XYZ_image.max() - XYZ_image.min())
    (x1, y1), (x2, y2) = spectralon_coords
    white_XYZ = XYZ_image[y1:y2, x1:x2, :].reshape(-1, 3).mean(axis=0)

    # von Kries: cada canal escalado pelo branco, D65 como referência
    XYZ_norm = XYZ_image / white_XYZ * np.array([0.95047, 1.0, 1.08883])
    XYZ_image = np.clip(XYZ_norm, 0, None)  # evita valores negativos

    sRGB_raw = colour.XYZ_to_sRGB(XYZ_image)
    fora = ((sRGB_raw < 0) | (sRGB_raw > 1)).any(axis=2)

    # Convert to sRGB
    sRGB = colour.XYZ_to_sRGB(XYZ_image)
    # Limit to [0, 1]
    sRGB = np.clip(sRGB, 0, 1)

    # Extract spectralon region in the image
    (x1, y1), (x2, y2) = spectralon_coords
    white_patch = sRGB[y1:y2, x1:x2, :]
    avg_white = np.mean(white_patch.reshape(-1, 3), axis=0)

    # Apply white balancing
    true_white = np.array([243.0, 243.0, 242.0]) / 255.0
    scale_factors = true_white / avg_white

    balanced = np.clip(sRGB * scale_factors, 0, 1)      # limits to [0, 1]

    # rgb_img = (balanced * 255).clip(0, 255).astype(np.uint8)
    rgb_img = (balanced * 255).clip(0, 255).astype(np.uint8)
    # self.main.cube_builder.bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    return rgb_img, XYZ_image



# Refiz a função de clusterização usando GMM
def cluster_image_gmm(X, n_clusters):

    #lab = cv2.cvtColor(X, cv2.COLOR_RGB2LAB)
    pixels = X.reshape(-1, 3)

    #testando sem o L
    #pixels = np.delete(pixels2, 0, axis=1)
    # não ficou legal

    gmm = GaussianMixture(n_components=n_clusters, random_state=0)

    gmm.fit(pixels)

    probab = gmm.predict_proba(pixels)
    means = gmm.means_ 
    idx_papel = np.argmax(means[:, 0])  # O cluster com o maior valor médio de L é provavelmente o papel
    probab = np.delete(probab, idx_papel, axis=1)
    mask = (probab > 0.9).astype(np.uint8)

    #labels = gmm.predict(pixels).reshape(X.shape[0], X.shape[1])
    
    return mask, means


def threshold_L(lab, T):
    L_channel = lab[:, :, 0]
    threshold = T  # valor de L abaixo do qual consideramos tinta
    mask = (L_channel < threshold).astype(np.uint8)
    return mask

def midpoint_cut_by_L(L_vals, threshold=15.0, min_group_size=30):
    """
    Divide os valores de L* cortando no Ponto Médio do intervalo (Midpoint Cut).
    Ideal para isolar pigmentos raros sem que a massa dominante desloque o corte.
    """
    def _split(indices):
        # 1. Se o grupo for menor que o mínimo para dividir em 2
        if len(indices) < min_group_size * 2:
            return [indices]
        
        vals = L_vals[indices]

        p_min = np.percentile(vals, 5)
        p_max = np.percentile(vals, 95)
        amplitude = p_max - p_min
        
        # Se a variação total de L* for pequena, é a mesma cor/tinta
        if amplitude <= threshold:
            return [indices]
        
        # 3. CORTE NO PONTO MÉDIO DO ESPETRO/INTERVALO
        ponto_medio_L = (p_min + p_max) / 2.0
        
        esq_mask = vals < ponto_medio_L
        dir_mask = vals >= ponto_medio_L
        
        # 4. Trava de segurança: garante que o grupo menor tem tamanho mínimo
        if esq_mask.sum() < min_group_size or dir_mask.sum() < min_group_size:
            return [indices]
            
        esquerda = indices[esq_mask]
        direita = indices[dir_mask]
        
        return _split(esquerda) + _split(direita)

    todos_indices = np.arange(len(L_vals))
    return _split(todos_indices)

def compute_ciede2000_matrix(features_lab):
    """
    features_lab: array (N, 3) com valores L*, a*, b* REAIS (não escalados)
    retorna matriz (N, N) de distâncias CIEDE2000
    """
    n = len(features_lab)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        # calcula delta_E entre o ponto i e todos os outros de uma vez
        dist_matrix[i] = colour.delta_E(
            features_lab[i], features_lab, method='CIE 2000'
        )
    
    return dist_matrix

def refine_by_luminosity(labels, features, threshold=12.0, min_group_size=3):
    """labels: (n_superpixels,) do HDBSCAN. features: (n_superpixels, 3) médias LAB."""
    new_labels = labels.copy()
    next_label = labels.max() + 1

    for l in np.unique(labels[labels >= 0]):
        idx = np.flatnonzero(labels == l)
        if len(idx) < min_group_size * 2:
            continue

        L = features[idx, 0]
        if np.percentile(L, 95) - np.percentile(L, 5) <= threshold:
            continue

        t = threshold_otsu(L)
        above, below = idx[L >= t], idx[L < t]
        if min(len(above), len(below)) < min_group_size:
            continue

        new_labels[above] = next_label
        next_label += 1

    return new_labels

def refine_by_chroma(labels, features, threshold=15.0, min_group_size=3):
    new_labels = labels.copy()
    next_label = labels.max() + 1

    for l in np.unique(labels[labels >= 0]):
        idx = np.flatnonzero(labels == l)
        if len(idx) < min_group_size * 2:
            continue

        C = np.hypot(features[idx, 1], features[idx, 2])
        if np.percentile(C, 95) - np.percentile(C, 5) <= threshold:
            continue

        t = threshold_otsu(C)
        above, below = idx[C >= t], idx[C < t]
        if min(len(above), len(below)) < min_group_size:
            continue

        new_labels[above] = next_label
        next_label += 1

    return new_labels

from sklearn.cluster import AgglomerativeClustering
def refine_by_ciede(labels, features, threshold=5.0, percentil=98, min_group_size=3):
    new_labels = labels.copy()
    next_label = labels.max() + 1

    for l in np.unique(labels[labels >= 0]):
        idx = np.flatnonzero(labels == l)
        if len(idx) < min_group_size * 2:
            continue

        sub_features = features[idx]
        dist_matrix = compute_ciede2000_matrix(sub_features)

        # só a metade superior da matriz (sem a diagonal, sem repetir pares)
        iu = np.triu_indices_from(dist_matrix, k=1)
        distancias = dist_matrix[iu]

        if np.percentile(distancias, percentil) <= threshold:
            continue

        clusterer = AgglomerativeClustering(
            n_clusters=2, metric='precomputed', linkage='average'
        )
        sub_labels = clusterer.fit_predict(dist_matrix)

        above = idx[sub_labels == 0]
        below = idx[sub_labels == 1]
        if min(len(above), len(below)) < min_group_size:
            continue

        new_labels[above] = next_label
        next_label += 1

    return new_labels

from sklearn.cluster import SpectralClustering

def projetar_por_ciede(sub_features, centro_a, centro_b):
    d_a = colour.delta_E(sub_features, centro_a, method='CIE 2000')
    d_b = colour.delta_E(sub_features, centro_b, method='CIE 2000')
    return d_b - d_a


def refine_by_ciede_spectral(labels, features, threshold=18.0, percentil=98,
                               min_group_size=3, sigma=None, assign_labels='discretize'):
    new_labels = labels.copy()
    next_label = labels.max() + 1

    for l in np.unique(labels[labels >= 0]):
        idx = np.flatnonzero(labels == l)
        if len(idx) < min_group_size * 2:
            continue

        sub_features = features[idx]
        dist_matrix = compute_ciede2000_matrix(sub_features)

        iu = np.triu_indices_from(dist_matrix, k=1)
        distancias = dist_matrix[iu]

        # mesmo critério de disparo que você já usa e já validou
        if np.percentile(distancias, percentil) <= threshold:
            continue

        # sigma robusto: mediana das distâncias par-a-par do próprio cluster
        # (reaproveita 'distancias', que já foi calculado acima)
        sigma_local = sigma if sigma is not None else np.median(distancias)
        if sigma_local <= 1e-6:
            continue  # cluster degenerado (quase todos os pontos idênticos)

        # converte distância em afinidade via kernel gaussiano
        affinity = np.exp(-(dist_matrix ** 2) / (2 * sigma_local ** 2))

        try:
            clusterer = SpectralClustering(
                n_clusters=2,
                affinity='precomputed',
                assign_labels=assign_labels,  # 'discretize' é determinístico
                random_state=0
            )
            sub_labels = clusterer.fit_predict(affinity)
        except Exception as e:
            print(f'Spectral falhou no cluster {l} (n={len(idx)}): {e}')
            continue

        above = idx[sub_labels == 0]
        below = idx[sub_labels == 1]
        if min(len(above), len(below)) < min_group_size:
            continue

        new_labels[above] = next_label
        next_label += 1

    return new_labels

def refine_recursive(labels, features, refine_fn, threshold, min_group_size=3, max_iter=10, percentil=98, alpha=0.02):
    for _ in range(max_iter):
        novos = refine_fn(labels, features, threshold=threshold, min_group_size=min_group_size, percentil=percentil, alpha=alpha)
        if (novos == labels).all():
            break
        labels = novos
    return labels

import diptest  # pip install diptest

def projetar_por_ciede(sub_features, centro_a, centro_b):
    """Projeta cada ponto numa reta 1D baseada na diferença de distância
    perceptual (CIEDE2000) até os dois centros candidatos, em vez de
    produto escalar Euclidiano em Lab bruto."""
    d_a = colour.delta_E(sub_features, centro_a, method='CIE 2000')
    d_b = colour.delta_E(sub_features, centro_b, method='CIE 2000')
    return d_b - d_a  # positivo = mais perto de A; negativo = mais perto de B


def refine_by_ciede_hybrid(labels, features, threshold=18.0, percentil=98,
                             min_group_size=3, alpha=0.02, sigma=None):
    new_labels = labels.copy()
    next_label = labels.max() + 1

    for l in np.unique(labels[labels >= 0]):
        idx = np.flatnonzero(labels == l)
        if len(idx) < min_group_size * 2:
            continue

        sub_features = features[idx]
        dist_matrix = compute_ciede2000_matrix(sub_features)
        iu = np.triu_indices_from(dist_matrix, k=1)
        distancias = dist_matrix[iu]

        if np.percentile(distancias, percentil) <= threshold:
            continue

        # divide via grafo (spectral), em vez de agglomerative
        sigma_local = sigma if sigma is not None else np.median(distancias)
        if sigma_local <= 1e-6:
            continue
        affinity = np.exp(-(dist_matrix ** 2) / (2 * sigma_local ** 2))
        try:
            clusterer = SpectralClustering(n_clusters=2, affinity='precomputed',
                                             assign_labels='discretize', random_state=0)
            sub_labels = clusterer.fit_predict(affinity)
        except Exception:
            continue

        above = idx[sub_labels == 0]
        below = idx[sub_labels == 1]
        if min(len(above), len(below)) < min_group_size:
            continue

        # valida a divisão com o dip test (mesma projeção CIEDE2000 corrigida)
        centro_a = sub_features[sub_labels == 0].mean(axis=0)
        centro_b = sub_features[sub_labels == 1].mean(axis=0)
        projecao = np.sort(projetar_por_ciede(sub_features, centro_a, centro_b))

        _, p_value = diptest.diptest(projecao)
        if p_value > alpha:
            continue  # spectral propôs, dip test não confirmou -> não divide

        print(f'cluster {l}: n={len(idx)}, p_value={p_value:.2e}, DE entre grupos={colour.delta_E(centro_a, centro_b, method="CIE 2000"):.1f}')

        new_labels[above] = next_label
        next_label += 1

    return new_labels

def refine_by_ciede_hybriddois(labels, features, threshold=18.0, percentil=98,
                             min_group_size=3, alpha=0.02, sigma=None):
    new_labels = labels.copy()
    next_label = labels.max() + 1

    def _avaliar_candidato(idx, sub_features, sub_labels, metodo):
        above = idx[sub_labels == 0]
        below = idx[sub_labels == 1]
        if min(len(above), len(below)) < min_group_size:
            return None

        centro_a = sub_features[sub_labels == 0].mean(axis=0)
        centro_b = sub_features[sub_labels == 1].mean(axis=0)
        projecao = np.sort(projetar_por_ciede(sub_features, centro_a, centro_b))

        dip_stat, p_value = diptest.diptest(projecao)
        if p_value > alpha:
            return None

        # dentro de _avaliar_candidato, depois de calcular centro_a e centro_b:
        d_intra_a = colour.delta_E(sub_features[sub_labels == 0], centro_a, method='CIE 2000').mean()
        d_intra_b = colour.delta_E(sub_features[sub_labels == 1], centro_b, method='CIE 2000').mean()
        d_inter = colour.delta_E(centro_a, centro_b, method='CIE 2000')

        # separação relativa: quão longe estão os centros comparado à dispersão interna
        # (análogo a um índice de Davies-Bouldin invertido; maior = melhor separação)
        score = d_inter / (0.5 * (d_intra_a + d_intra_b) + 1e-9)

        return {
            'score': score,
            'metodo': metodo,
            'above': above,
            'below': below,
            'centro_a': centro_a,
            'centro_b': centro_b,
            'dip_stat': dip_stat,
            'p_value': p_value,
        }

    for l in np.unique(labels[labels >= 0]):
        idx = np.flatnonzero(labels == l)
        if len(idx) < min_group_size * 2:
            continue

        sub_features = features[idx]
        dist_matrix = compute_ciede2000_matrix(sub_features)
        iu = np.triu_indices_from(dist_matrix, k=1)
        distancias = dist_matrix[iu]

        if np.percentile(distancias, percentil) <= threshold:
            continue

        candidatos = []

        # --- candidato 1: hierárquico (average linkage) ---
        agglo = AgglomerativeClustering(n_clusters=2, metric='precomputed',
                                          linkage='average')
        sub_labels_agglo = agglo.fit_predict(dist_matrix)
        resultado = _avaliar_candidato(idx, sub_features, sub_labels_agglo, 'agglomerative')
        if resultado is not None:
            candidatos.append(resultado)

        # --- candidato 2: espectral (grafo de afinidade RBF) ---
        sigma_local = sigma if sigma is not None else np.median(distancias)
        if sigma_local > 1e-6:
            affinity = np.exp(-(dist_matrix ** 2) / (2 * sigma_local ** 2))
            try:
                spectral = SpectralClustering(n_clusters=2, affinity='precomputed',
                                                assign_labels='discretize', random_state=0)
                sub_labels_spectral = spectral.fit_predict(affinity)
                resultado = _avaliar_candidato(idx, sub_features, sub_labels_spectral, 'spectral')
                if resultado is not None:
                    candidatos.append(resultado)
            except Exception:
                pass

        if not candidatos:
            continue  # nem hierárquico nem espectral produziram uma divisão válida

        # --- desempate: maior estatística de dip vence ---
        vencedor = max(candidatos, key=lambda c: c['score'])

        print(f"cluster {l}: n={len(idx)}, método={vencedor['metodo']}, "
              f"dip={vencedor['dip_stat']:.4f}, p_value={vencedor['p_value']:.2e}, "
              f"DE entre grupos={colour.delta_E(vencedor['centro_a'], vencedor['centro_b'], method='CIE 2000'):.1f}")

        new_labels[vencedor['above']] = next_label
        next_label += 1

    return new_labels


def refine_by_ciede_diptest(labels, features, threshold=18.0, percentil=98,
                              min_group_size=3, alpha=0.1):
    new_labels = labels.copy()
    next_label = labels.max() + 1

    for l in np.unique(labels[labels >= 0]):
        idx = np.flatnonzero(labels == l)
        if len(idx) < min_group_size * 2:
            continue

        sub_features = features[idx]
        dist_matrix = compute_ciede2000_matrix(sub_features)
        iu = np.triu_indices_from(dist_matrix, k=1)
        distancias = dist_matrix[iu]

        # pré-filtro barato: só testa estatisticamente se já parece disperso
        if np.percentile(distancias, percentil) <= threshold:
            continue

        clusterer = AgglomerativeClustering(n_clusters=2, metric='precomputed', linkage='average')
        sub_labels = clusterer.fit_predict(dist_matrix)

        above = idx[sub_labels == 0]
        below = idx[sub_labels == 1]
        if min(len(above), len(below)) < min_group_size:
            continue

        # --- validação: a divisão proposta é estatisticamente real? ---
        centro_a = sub_features[sub_labels == 0].mean(axis=0)
        centro_b = sub_features[sub_labels == 1].mean(axis=0)

        projecao = np.sort(projetar_por_ciede(sub_features, centro_a, centro_b))

        _, p_value = diptest.diptest(projecao)
        if p_value > alpha:
            continue  # não rejeita unimodalidade -> mantém 1 cluster só

        new_labels[above] = next_label
        next_label += 1

    return new_labels

# Criei uma função para aplica PCA em uma matriz X tridimensional
def applypca(X):
    # Juntamos suas duas primeiras dimensões em uma só
    X_flat = X.flatten().reshape(-1, X.shape[-1])
    X_np = np.array(X_flat, dtype=np.float64)
    # Aplicamos PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit(X_np)
    X_pca_reshaped = X_pca.transform(X_np).reshape(X.shape[0], X.shape[1], 3)
    return(X_pca_reshaped)

def cluster_image(image_np):
    
    pixels = image_np.reshape(-1, 3)

    # mask = np.all(pixels != [0, 0, 0], axis=1)
    # pixels_valid = pixels[mask]
     # cluster_size = 60, min_samples = 4, cluster_selection_epsilon = 0.05, cluster_selection_method = 'eom'
    clusterer  = HDBSCAN(min_cluster_size=30, min_samples=4, cluster_selection_epsilon=0.1, cluster_selection_method='eom')

    group = clusterer.fit(pixels)

    # labels = np.full (len(pixels), -1)
    # labels[mask] = kmeans.labels_
    labels = group.labels_
    # labels = labels.reshape(image_np.shape[0], image_np.shape[1])
    
    return labels

from sklearn.cluster import HDBSCAN as HDBSCAN_CPU

def cluster_image_ciede2000(features_lab):
    dist_matrix = compute_ciede2000_matrix(features_lab)
    
    clusterer = HDBSCAN_CPU(
        min_cluster_size=40,
        min_samples=7,  #era 6
        cluster_selection_epsilon=1,  # era 1
        cluster_selection_method='eom',
        metric='precomputed'
    )
    
    labels = clusterer.fit_predict(dist_matrix)

    return labels, dist_matrix

def limpa_imagem(img_np, min_size):
    # Máscara binária: True onde o pixel não é preto
    mascara = img_np.any(axis=2)

    estrutura = np.ones((3, 3))
    rotulos, n = ndimage.label(mascara, structure=estrutura)

    # Conta tamanho de cada componente
    tamanhos = ndimage.sum(mascara, rotulos, range(1, n + 1))

    # Identifica componentes pequenos
    pequenos = np.where(np.array(tamanhos) <= min_size)[0] + 1

    # Zera os pixels dos componentes pequenos
    mascara_remover = np.isin(rotulos, pequenos)
    img_np[mascara_remover] = [0, 0, 0]


from scipy.spatial.distance import cdist

def merge_clusters_ciede2000(labels, features_lab, threshold=5.0):
    unique_labels = np.unique(labels[labels >= 0])
    
    centroids = []
    for l in unique_labels:
        mask = (labels == l)
        centroids.append(features_lab[mask].mean(axis=0))
    centroids = np.array(centroids)
    
    # matriz de distância CIEDE2000 entre centróides
    dists = compute_ciede2000_matrix(centroids)
    
    mapping = {l: l for l in unique_labels}
    for i in range(len(unique_labels)):
        for j in range(i+1, len(unique_labels)):
            if dists[i, j] < threshold:
                label_i = mapping[unique_labels[i]]
                label_j = mapping[unique_labels[j]]
                if label_i != label_j:
                    for k in mapping:
                        if mapping[k] == label_j:
                            mapping[k] = label_i
    
    new_labels = labels.copy()
    for old, new in mapping.items():
        new_labels[labels == old] = new
    
    new_labels_final = np.full_like(new_labels, -1)
    for new_id, label in enumerate(np.unique(new_labels[new_labels >= 0])):
        new_labels_final[new_labels == label] = new_id
    
    return new_labels_final

def filter_superpix(segments, lab_corrigido, mask_tinta):
    features = []
    indices_validos = []
    n_segments = segments.max() + 1
    descartados = []

    for i in range(n_segments):
        mask_seg = (segments == i)

        pixels_no_seg = mask_seg.sum()
        if pixels_no_seg == 0:
            continue

        pixels_tinta = (mask_seg & mask_tinta).sum()
        if pixels_tinta / pixels_no_seg < 0.5:
            continue

        pixels_seg = lab_corrigido[mask_seg & mask_tinta]
        if len(pixels_seg) < 0:          # era: == 8
            continue

        variancia = pixels_seg.var(axis=0).mean()
        if variancia > 15:
            descartados.append(i)
            continue

        mean_color = pixels_seg.mean(axis=0)
        features.append(mean_color)
        indices_validos.append(i)

    return features, indices_validos, n_segments, descartados

def gmm_L(lab, n_components=2):
    """Separa tinta de suporte via GMM em L* apenas.
    Retorna máscara uint8: 1 = tinta, 0 = suporte."""
    L = lab[:, :, 0].reshape(-1, 1)

    # subamostra para ajustar (a imagem inteira é lenta demais)
    amostra = L[::37]
    gmm = GaussianMixture(n_components=n_components, random_state=0).fit(amostra)

    idx_papel = np.argmax(gmm.means_.ravel())
    rotulos = gmm.predict(L)
    mask = (rotulos != idx_papel).reshape(lab.shape[:2])

    print('médias L* dos componentes:', np.sort(gmm.means_.ravel()).round(1))
    print(f'fração marcada como tinta: {100*mask.mean():.1f}%')
    return mask.astype(np.uint8)

def threshold_L_croma(lab, T_L=None, p_croma=98):
    """Tinta = escura OU cromática. Retorna máscara uint8."""
    L = lab[:, :, 0]
    C = np.hypot(lab[:, :, 1], lab[:, :, 2])

    if T_L is None:
        T_L = threshold_otsu(L)
    T_C = np.percentile(C, p_croma)

    mask = (L < T_L) | (C > T_C)

    print(f'T_L={T_L:.1f}, T_C={T_C:.1f}')
    print(f'só por L*: {100*(L<T_L).mean():.1f}%, '
          f'só por croma: {100*(C>T_C).mean():.1f}%, '
          f'união: {100*mask.mean():.1f}%')
    return mask.astype(np.uint8)

threshold = True
image_paths=[
    "imagens12/Antifonario_-_pag_246_verso_DataCube-365.tif",
    "imagens12/Antifonario_-_pag_246_verso_DataCube-385.tif",
    "imagens12/Antifonario_-_pag_246_verso_DataCube-395.tif",
    "imagens12/Antifonario_-_pag_246_verso_DataCube-420.tif",
    "imagens12/Antifonario_-_pag_246_verso_DataCube-450.tif",
    "imagens12/Antifonario_-_pag_246_verso_DataCube-470.tif",
    "imagens12/Antifonario_-_pag_246_verso_DataCube-500.tif",
    "imagens12/Antifonario_-_pag_246_verso_DataCube-530.tif",
    "imagens12/Antifonario_-_pag_246_verso_DataCube-560.tif",
    "imagens12/Antifonario_-_pag_246_verso_DataCube-590.tif",
    "imagens12/Antifonario_-_pag_246_verso_DataCube-615.tif",
    "imagens12/Antifonario_-_pag_246_verso_DataCube-630.tif",
    "imagens12/Antifonario_-_pag_246_verso_DataCube-660.tif",
    "imagens12/Antifonario_-_pag_246_verso_DataCube-730.tif",
    "imagens12/Antifonario_-_pag_246_verso_DataCube-850.tif",
    "imagens12/Antifonario_-_pag_246_verso_DataCube-940.tif"
]

T_TINTA = 72    #era 72, depois foi 67
PUREZA_MIN = 0.95
N_SEGMENTS = 20000
CACHE = f"cache_T{T_TINTA}_pureza{PUREZA_MIN}_seg{N_SEGMENTS}.npz"

if os.path.exists(CACHE):
    d = np.load(CACHE)
    segments        = d['segments']
    features        = d['features']
    indices_validos = d['indices_validos']
    n_segments      = int(d['n_segments'])
    mask_tinta      = d['mask_tinta']
    lab_corrigido   = d['lab_corrigido'].astype(np.float64)
    img_np          = d['img_np'].copy()
    descartados     = d['descartados']
    print(f"cache carregado: {len(features)} superpixels válidos")

else:

    images = []

    for i in range(len(image_paths)):
        print(i)
        img = Image.open(image_paths[i]).convert('L')
        # if (i==14):
        #     img = img.transpose(Image.ROTATE_270)
        img = np.array(img)
        images.append(img)
        print(images[i].shape)
    X = np.array(images)



    #colore    
    spectralon_coords = ((20, 1357), (150, 1511))
    sd = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    print(sd.shape)
    rgb_array, xyz_array = generate_true_color_image(X, [365, 385, 395, 420, 450, 470, 500, 530, 560, 590, 615, 630, 660, 730, 850, 940], spectralon_coords)
    rgb_image =  Image.fromarray(rgb_array)
    rgb_image.save("colorido.png")
    imagecropped = rgb_image.crop((0, 400, 5467, 3643))
    imagecropped.save("cropped.png")

    # #separa papel
    # imagecropped = Image.open("imagens6/241216_antifonario_folha246_verso_200_RGB.tif")

    n_clusters = 2
    img_np = np.array(imagecropped)

    img_float = img_np.astype(float) / 255.0
    lab = xyz2lab(xyz_array)
    #lab = rgb2lab(img_float)

    print(lab[:,:,0].min(), lab[:,:,0].max(), lab[:,:,0].mean())
    # no spectralon:
    print(lab[1304:1559, 2:183, 0].mean())

    lab = lab[400:3643, 0:5467]



    mask = threshold_L_croma(lab, T_L=T_TINTA, p_croma=PUREZA_MIN*100)
    

    # labels, means = cluster_image_gmm(lab, n_clusters)
    # labels = labels.reshape(img_np.shape[0], img_np.shape[1])

    # mask = [(labels == i).astype(np.uint8) * 255 for i in range(n_clusters)]

    # separado = []

    # for i in range(n_clusters):
    #     # cria máscara
    #     mask3 = mask[i][:, :, None] 
    #     cluster_img = np.where(mask3 == 255, img_np, 0).astype(np.uint8)
    #     separado.append(cluster_img)

    # score = means[:, 0] + means[:, 2]
    # idx_papel = np.argmax(score)

    # mask = threshold_L(lab, T=72)

    mask = threshold_L_croma(lab, T_L=T_TINTA, p_croma=PUREZA_MIN*100)

    mask_papel = ~mask.astype(bool)

    cor_papel_ab = lab[mask_papel][:, 1:].mean(axis=0)  # média de a* e b* do papel

    # subtrai o cast de amarelamento só nos canais de cor
    lab_corrigido = lab.copy()
    # tirei p testar
    # lab_corrigido[:, :, 1] -= cor_papel_ab[0]  # corrige a*
    # lab_corrigido[:, :, 2] -= cor_papel_ab[1]  # corrige b*

    tinta = np.where(mask[:, :, None], img_np, 0).astype(np.uint8)


    # Não padronizamos toda a imagem globalmente aqui; vamos normalizar apenas os superpixels depois.
    Image.fromarray(tinta).save("tinta.png")

    # mask = threshold_L_croma(lab)
    # mask_papel = ~mask.astype(bool)

    #clusteriza

    img_np = tinta
    mask = np.any(img_np != [0, 0, 0], axis=2)
    lab_tinta = lab.copy()
    lab_tinta[~mask] = 0  # zera o fundo
    #lab = xyz2lab(img_np)
    print("começou superpixel")
    segments = slic(lab_tinta, n_segments=20000, mask=mask, compactness=3)
    print("terminou superpixel")

    mask_tinta = np.any(img_np != [0, 0, 0], axis=2)  # pixels não-pretos
    #print(f"pixels fora do gamut: {100*fora[mask_tinta].mean():.2f}% da tinta")

    features, indices_validos, n_segments, descartados = filter_superpix(segments, lab_corrigido, mask_tinta)

    if len(features) == 0:
        print('Nenhum superpixel válido com os filtros atuais. Fazendo fallback usando todos os superpixels com tinta.')
        for i in range(n_segments):
            mask_seg = (segments == i)
            pixels_seg = lab_corrigido[mask_seg & mask_tinta]
            if len(pixels_seg) == 0:
                continue
            mean_color = pixels_seg.mean(axis=0)
            features.append(mean_color)
            indices_validos.append(i)

    print(f"Total superpixels: {n_segments}, Válidos para clusterização: {len(features)}")
    features = np.array(features, dtype=float)
    indices_validos = np.array(indices_validos, dtype=int)

    if features.ndim != 2 or features.shape[1] != 3:
        raise ValueError(f'Features inválidas: shape={features.shape}')

    np.savez_compressed(
        CACHE,
        segments=segments, features=features,
        indices_validos=indices_validos, n_segments=n_segments,
        mask_tinta=mask_tinta,
        lab_corrigido=lab_corrigido.astype(np.float32),
        img_np=img_np,
        descartados=np.array(descartados, dtype=int)
    )

# Usamos apenas a* e b* para clusterização, minimizando efeitos de iluminação e papel.
features_scaled = features.copy()

# features = []
# for i in range(n_segments):
#     mask2 = (segments == i)
#     mean_color = lab[mask2].mean(axis=0)
#     features.append(mean_color)

# features = np.array(features)


def varre_epsilon(features_scaled, features_lab, epsilons=[0.1, 0.2, 0.3]):
    resultados = []
    for eps in epsilons:

        cl = HDBSCAN(
            min_cluster_size=30,
            min_samples=4,
            cluster_selection_epsilon=eps,
            cluster_selection_method='eom',
        )
        labels = cl.fit_predict(features_scaled)
        n_clusters = len(np.unique(labels[labels >= 0]))
        n_ruido = (labels == -1).sum()
        if(threshold):
            for t in [1.0, 5.0, 10.0, 15.0]:
                labels = merge_clusters_ciede2000(labels, features_lab, threshold=t)
                n_clusters = len(np.unique(labels[labels >= 0]))
                n_ruido = (labels == -1).sum()
                resultados.append((eps, t, n_clusters, n_ruido))
                print(f"epsilon={eps:.2f}, threshold={t:.2f} → {n_clusters} clusters, {n_ruido} ruído")
        else:
            for eps2 in [0.01, 0.05, 0.1, 0.15]:
                new_labels = refine_by_luminosity(
                                labels_hue=old_labels,
                                features=features,
                                threshold=40,
                                min_cluster_size=20,
                                min_samples=3,
                                cluster_selection_epsilon=eps2,
                            )
                n_clusters = len(np.unique(new_labels[new_labels >= 0]))
                n_ruido = (new_labels == -1).sum()
                resultados.append((eps, n_clusters, n_ruido))
                print(f"epsilon={eps:.2f} → epsilon2={eps2:.2f} → {n_clusters} clusters, {n_ruido} ruído")
    return resultados

# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(features)

def test_scalers_and_run(features):
    print('\n--- Estatísticas das features (antes do scaler) ---')
    print('shape:', features.shape)
    print('mean:', np.mean(features, axis=0))
    print('std: ', np.std(features, axis=0))

    experiments = {
        #'none_full_LAB': features.copy(),
        'standard_full_LAB': StandardScaler().fit_transform(features),
        # 'standard_ab': StandardScaler().fit_transform(features[:, 1:]),
        # 'minmax_ab': MinMaxScaler().fit_transform(features[:, 1:]),
        # 'robust_ab': RobustScaler().fit_transform(features[:, 1:])
    }

    for name, feat in experiments.items():
        print(f"\n--- Teste scaler: {name} ---")
        print('shape:', feat.shape)
        # imprime estatísticas por canal (ou por dimensão)
        print('mean:', np.mean(feat, axis=0))
        print('std: ', np.std(feat, axis=0))

        feat_input = feat.copy()

        # Ajusta epsilons adequados (varre_epsilon imprimirá resultados)
        try:
            varre_epsilon(feat_input, features)
        except Exception as e:
            print('Erro ao rodar HDBSCAN com', name, '→', e)


# if features.size > 0:
#     test_scalers_and_run(features)

# print("Resultados para diferentes valores de epsilon:")
# resultados = varre_epsilon(features_scaled,)




tinta = np.where(mask_tinta[:, :, None], img_np, 0).astype(np.uint8)

old_labels, dist_matrix = cluster_image_ciede2000(features_scaled)

usar_merge = False  # se True, faz merge de clusters próximos em CIEDE2000

if usar_merge:
    old_labels = merge_clusters_ciede2000(old_labels, features, threshold=7.0)

# --- imagens de "antes" ---
lookup = np.full(n_segments, -1)
lookup[indices_validos] = old_labels
cluster_map = lookup[segments]

for i in range(old_labels.max() + 1):
    mask3 = ((cluster_map == i).astype(np.uint8) * 255)[:, :, None]
    cluster_img = np.where(mask3 == 255, img_np, 0).astype(np.uint8)
    limpa_imagem(cluster_img, min_size=15)
    if cluster_img.max() > 0:
        Image.fromarray(cluster_img).save(f"limpo_pre_{i}.png")
    c = features[old_labels == i]
    print(f"cluster {i}: n={len(c)}")
    print(f"   L* p5/50/95: {np.percentile(c[:,0],[5,50,95]).round(1)}")
    print(f"   a* p5/50/95: {np.percentile(c[:,1],[5,50,95]).round(1)}")
    print(f"   b* p5/50/95: {np.percentile(c[:,2],[5,50,95]).round(1)}")



# L_ = 3  # o número que você achou
# idx = np.flatnonzero(old_labels == L_)
# plt.hist(features[idx, 2], bins=60)
# plt.xlabel('b*'); plt.savefig('b_dentro.png')
# print('n =', len(idx))
# print('percentis b*:', np.percentile(features[idx,2], [1,5,25,50,75,95,99]).round(1))

# --- refinamento ---
# labels = refine_recursive(old_labels, features, refine_by_chroma, threshold=30.0)
# labels = refine_recursive(labels, features, refine_by_luminosity, threshold=20.0)
# labels = refine_recursive(old_labels, features, refine_by_ciede_spectral, threshold=18, min_group_size=30, percentil=98) #era 18
labels = refine_recursive(old_labels, features, refine_by_ciede_hybriddois, threshold=18, min_group_size=30, percentil=98 ,alpha=1e-5) #era 18

print('clusters encontrados:', np.unique(labels, return_counts=True))

lookup = np.full(n_segments, -1)
lookup[indices_validos] = labels
cluster_map = lookup[segments]

for i in range(labels.max() + 1):
    mask3 = ((cluster_map == i).astype(np.uint8) * 255)[:, :, None]
    cluster_img = np.where(mask3 == 255, img_np, 0).astype(np.uint8)
    limpa_imagem(cluster_img, min_size=15)
    if cluster_img.max() > 0:
        Image.fromarray(cluster_img).save(f"limpo_{i}.png")
    c = features[labels == i]
    print(f"cluster {i}: n={len(c)}")
    print(f"   L* p5/50/95: {np.percentile(c[:,0],[5,50,95]).round(1)}")
    print(f"   a* p5/50/95: {np.percentile(c[:,1],[5,50,95]).round(1)}")
    print(f"   b* p5/50/95: {np.percentile(c[:,2],[5,50,95]).round(1)}")

def centroides_de(labels, features):
    return {k: features[labels == k].mean(axis=0) for k in np.unique(labels[labels >= 0])}

centroides = centroides_de(labels, features)

lookup = np.full(n_segments, -1)
lookup[indices_validos] = labels

cont = 0
for i in descartados:
    cor = np.mean(lab_corrigido[(segments == i) & mask_tinta], axis=0)
    d = {k: colour.delta_E(cor, cent, method='CIE 2000')
        for k, cent in centroides.items()}
    k_prox = min(d, key=d.get)
    if d[k_prox] < 20:
        lookup[i] = k_prox
        cont+=1
print(f"Total superpixels: {n_segments}, Válidos para clusterização: {len(features)}")
print(f'{cont} superpixels descartados foram reatribuídos a clusters próximos')

cluster_map = lookup[segments]

clusters_images = []
clusters_lab_images = []

for i in range(labels.max()+1):
    mask3 = ((cluster_map == i).astype(np.uint8) * 255)[:, :, None]
    cluster_img = np.where(mask3 == 255, img_np, 0).astype(np.uint8)
    cluster_lab = np.where(mask3 == 255, lab_corrigido * 255, 0).astype(np.uint8)
    clusters_images.append(cluster_img)
    clusters_lab_images.append(cluster_lab)

#limpa


for cont, img_np in enumerate(clusters_images):
    limpa_imagem(img_np, min_size=15)
    todos_zero = img_np.max() == 0
    if (not todos_zero):
        Image.fromarray(img_np.astype(np.uint8)).save(f"limpo_pos_{cont}.png")

# for cont, img_np in enumerate(clusters_lab_images):
#     limpa_imagem(img_np, min_size=15)
#     todos_zero = img_np.max() == 0
#     if (not todos_zero):
#         Image.fromarray(img_np.astype(np.uint8)).save(f"limpo_lab_{cont}.png")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(features[:, 1], features[:, 2], alpha=0.3, s=1)
axes[0].set_xlabel('a*'); axes[0].set_ylabel('b*')
axes[0].set_title('a* vs b*')

axes[1].scatter(features[:, 0], features[:, 1], alpha=0.3, s=1)
axes[1].set_xlabel('L*'); axes[1].set_ylabel('a*')
axes[1].set_title('L* vs a*')

axes[2].scatter(features[:, 0], features[:, 2], alpha=0.3, s=1)
axes[2].set_xlabel('L*'); axes[2].set_ylabel('b*')
axes[2].set_title('L* vs b*')

plt.tight_layout()
plt.savefig('distribuicao_features.png')

# salva a imagem do lab_corrigido para ver se o pré-processamento faz sentido visualmente
lab_viz = (lab_corrigido * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(lab_viz).save("lab_corrigido_viz.png")

# e também o SLIC para ver como os superpixels ficaram
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

fig, ax = plt.subplots(figsize=(15, 10))
ax.imshow(mark_boundaries(img_as_float(tinta), segments))
plt.savefig("superpixels_viz.png", dpi=150)

# quantas distâncias entre superpixels de cores "diferentes" existem?
from scipy.spatial.distance import pdist

distancias = pdist(features_scaled)
print(f"Distância mínima: {distancias.min():.2f}")
print(f"Distância média: {distancias.mean():.2f}")
print(f"Distância máxima: {distancias.max():.2f}")
print(f"Percentil 10: {np.percentile(distancias, 10):.2f}")
print(f"Percentil 90: {np.percentile(distancias, 90):.2f}")

# histograma das distâncias
plt.figure(figsize=(10, 4))
plt.hist(distancias, bins=100)
plt.xlabel('Distância entre superpixels')
plt.title('Distribuição de distâncias')
plt.savefig('histograma_distancias.png')

l = 2  # o cluster do ornamento vermelho
idx = np.flatnonzero(old_labels == l)

L_super = features[idx, 0]
L_pix = lab_corrigido[np.isin(segments, indices_validos[idx]) & mask_tinta][:, 0]

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(L_super, bins=50); ax[0].set_title('L* por superpixel')
ax[1].hist(L_pix, bins=100);  ax[1].set_title('L* por pixel')
plt.savefig('bimodal.png')