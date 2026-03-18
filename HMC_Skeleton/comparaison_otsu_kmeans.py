import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def main():
    # Chemins des images
    img_path = 'sources/cible_64_bruit.png'
    if not os.path.exists(img_path):
        img_path = 'Peano/sources/cible_64_bruit.png'
        
    hmc_res_path = 'results/cible_64_bruit_segmentation_result.png'
    
    # 1. Lecture de l'image originale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erreur: Image source introuvable.")
        return

    # 2. Otsu
    ret, th_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. K-Means
    # Reshape pour k-means (1D array de pixels)
    img_flat = img.reshape((-1, 1)).astype(np.float32)
    # Criteres k-means: (type, max_iter, epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    K = 2
    _, labels, centers = cv2.kmeans(img_flat, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Reconstruire l'image segmentee
    centers = np.uint8(centers)
    res_kmeans = centers[labels.flatten()]
    th_kmeans = res_kmeans.reshape(img.shape)

    # Afficher et sauvegarder la comparaison
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Image originale (bruitée)')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(th_otsu, cmap='gray')
    plt.title("Seuillage d'Otsu")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(th_kmeans, cmap='gray')
    plt.title('Clustering K-Means (K=2)')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    if os.path.exists(hmc_res_path):
        img_hmc = cv2.imread(hmc_res_path, cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_hmc, cmap='gray')
    else:
        plt.text(0.5, 0.5, 'HMC result introuvable', ha='center', va='center')
    plt.title('Modèle de Markov (HMC + Peano)')
    plt.axis('off')
    
    plt.tight_layout()
    out_path = 'results/cible_64_bruit_comparison.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"Comparaison sauvegardée dans {out_path}")

if __name__ == '__main__':
    main()
