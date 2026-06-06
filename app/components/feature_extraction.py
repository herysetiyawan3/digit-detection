import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA

def extract_features_pca(X, n_components=None):
    X_flattened = np.array([img.flatten() for img in X])  # Meratakan gambar
    # Gunakan n_components yang diberikan, jika tidak ada gunakan default 0.95
    pca = PCA(n_components=n_components if n_components is not None else 0.95, random_state=42)
    features = pca.fit_transform(X_flattened)
    return features


# Fungsi untuk ekstraksi fitur dengan HOG
def extract_features_hog(X):
    features = []
    for img in X:
        fd, _ = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        features.append(fd)
    features = np.array(features)
    return features

# Fungsi untuk memutuskan metode ekstraksi berdasarkan pilihan pengguna
def extract_features(X, method, n_components=None):
    if method == "PCA":
        return extract_features_pca(X, n_components)  # Teruskan n_components ke extract_features_pca
    elif method == "HOG":
        return extract_features_hog(X)  # Tidak memerlukan n_components untuk HOG
    else:
        raise ValueError("Metode ekstraksi fitur tidak valid!")

