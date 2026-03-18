import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

img_path = 'sources/cible_64_bruit.png'
if not os.path.exists(img_path):
    img_path = 'Peano/sources/cible_64_bruit.png'

hmc_path = 'results/cible_64_bruit_segmentation_result.png'

# Load original noisy image
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 1. Create Virtual Ground Truth (strong blur + Otsu to get smooth true shapes)
img_blur = cv2.GaussianBlur(img, (9, 9), 0)
_, gt = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
gt_binary = (gt > 127).astype(int)

# 2. Otsu
_, otsu_th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
otsu_binary = (otsu_th > 127).astype(int)

# 3. K-Means
img_flat = img.reshape((-1, 1)).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(img_flat, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
if centers[0] > centers[1]:
    kmeans_binary = (labels.flatten() == 0).astype(int)
else:
    kmeans_binary = (labels.flatten() == 1).astype(int)
kmeans_binary = kmeans_binary.reshape(img.shape)

# 4. HMC
import sys
sys.path.append('./Peano')
from PeanoImage import Peano
from InvPeanoImage import PeanoInverse
from func import InitParam, getProbaMarkov, getAlpha, getBeta, getGamma, EM_Iter, getMPMClassif

Y = Peano(img.astype(float))
nbIter = 30
K = 2
meanTabIter = np.zeros(shape=(nbIter, K))
varTabIter  = np.zeros(shape=(nbIter, K))
cTabIter    = np.zeros(shape=(nbIter, K, K))
tTabIter    = np.zeros(shape=(nbIter, K, K))
ITabIter    = np.zeros(shape=(nbIter, K))

iteration = 0
meanTabIter[iteration, :], varTabIter[iteration, :], cTabIter[iteration, :, :] = InitParam(K, Y)
tTabIter[iteration, :, :], ITabIter[iteration, :] = getProbaMarkov(cTabIter[iteration, :, :])

alpha, S = getAlpha(Y, meanTabIter[iteration, :], varTabIter[iteration, :], ITabIter[iteration, :], tTabIter[iteration, :, :])
beta     = getBeta(Y, meanTabIter[iteration, :], varTabIter[iteration, :], ITabIter[iteration, :], tTabIter[iteration, :, :], S)
gamma    = getGamma(alpha, beta)

for iteration in range(1, nbIter):
    gamma = EM_Iter(iteration, Y, meanTabIter, varTabIter, cTabIter, tTabIter, ITabIter)

X_MPM = getMPMClassif(gamma)
segmented_image = PeanoInverse(X_MPM)
hmc_binary = (segmented_image == 1).astype(int)
# match mapping to gt
if np.sum(hmc_binary == gt_binary) < np.sum((1-hmc_binary) == gt_binary):
    hmc_binary = 1 - hmc_binary

# 5. Evaluate Accuracies
# Handle potential class inversion naturally by matching to highest overlap
def get_acc(pred, true):
    acc1 = np.sum(pred == true)
    acc2 = np.sum((1-pred) == true)
    return max(acc1, acc2)

N = img.size
otsu_correct = get_acc(otsu_binary, gt_binary)
kmeans_correct = get_acc(kmeans_binary, gt_binary)
hmc_correct = get_acc(hmc_binary, gt_binary)

otsu_err = N - otsu_correct
kmeans_err = N - kmeans_correct
hmc_err = N - hmc_correct

# 6. Plotting
labels_plot = ['Otsu', 'K-Means', 'HMC (Markov)']
corrects = [otsu_correct, kmeans_correct, hmc_correct]
errors = [otsu_err, kmeans_err, hmc_err]

x = np.arange(len(labels_plot))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, corrects, width, label='Bien classés (Corrects)', color='g')
rects2 = ax.bar(x + width/2, errors, width, label='Mal classés (Erreurs)', color='r')

ax.set_ylabel('Nombre de pixels')
ax.set_title('Qualité de la Segmentation (Base: Vérité Terrain sur 4096 px)')
ax.set_xticks(x)
ax.set_xticklabels(labels_plot)
# Place legend outside to avoid overlap with bars
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1))

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
out_path = 'results/cible_64_bruit_accuracy_graph.png'
plt.savefig(out_path, dpi=150)
print(f"Graphique sauvegardé dans {out_path}")
