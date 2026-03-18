import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from func import *
from os.path import join
from PIL import Image

sys.path.append('./Peano')
from PeanoImage import Peano
from InvPeanoImage import PeanoInverse

if __name__ == '__main__':

    reposource = './sources'
    peano_source = './Peano/sources'
    reporesult = './results'
    
    # Load and process the source image
    input_image = sys.argv[1] if len(sys.argv) > 1 else 'cible_64_bruit.png'
    candidate_paths = [
        input_image,
        join(reposource, input_image),
        join(peano_source, input_image),
    ]
    imagename = None
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            imagename = candidate
            break
    if imagename is None:
        raise FileNotFoundError(f'Image not found: {input_image}')
    image_stem = os.path.splitext(os.path.basename(imagename))[0]
    # Load as float directly since Peano expects it
    image_orig = np.array(Image.open(imagename), dtype=float)
    
    # 1. Convert to 1D vector using Peano curve
    print("Applying Peano transform...")
    Y = Peano(image_orig)
    
    N = Y.shape[0]
    K = 2 # Assuming 2 classes (shape vs background)
    nbIter = 30
    
    # Parameters arrays
    meanTabIter = np.zeros(shape=(nbIter, K))
    varTabIter  = np.zeros(shape=(nbIter, K))
    cTabIter    = np.zeros(shape=(nbIter, K, K))
    tTabIter    = np.zeros(shape=(nbIter, K, K))
    ITabIter    = np.zeros(shape=(nbIter, K))
    
    # 2. EM algorithm to estimate parameters non-supervised
    print("Starting EM iterations...")
    iteration = 0
    meanTabIter[iteration, :], varTabIter[iteration, :], cTabIter[iteration, :, :] = InitParam(K, Y)
    tTabIter[iteration, :, :], ITabIter[iteration, :] = getProbaMarkov(cTabIter[iteration, :, :])

    alpha, S = getAlpha(Y, meanTabIter[iteration, :], varTabIter[iteration, :], ITabIter[iteration, :], tTabIter[iteration, :, :])
    beta     = getBeta(Y, meanTabIter[iteration, :], varTabIter[iteration, :], ITabIter[iteration, :], tTabIter[iteration, :, :], S)
    gamma    = getGamma(alpha, beta)

    for iteration in range(1, nbIter):
        if iteration % 5 == 0:
            print(f"---> EM iteration= {iteration}")
        
        gamma = EM_Iter(iteration, Y, meanTabIter, varTabIter, cTabIter, tTabIter, ITabIter)

    # 3. MPM Classification
    print("Applying MPM classification...")
    X_MPM = getMPMClassif(gamma)
    
    # Calculate the restored signal values by mapping class to the estimated means
    final_means = meanTabIter[nbIter-1, :]
    restored_signal = np.zeros(N)
    for n in range(N):
        restored_signal[n] = final_means[int(X_MPM[n])]
        
    # 4. Inverse Peano transform to get segmented Image
    print("Applying Inverse Peano transform...")
    segmented_image = PeanoInverse(restored_signal)
    
    # Save the original and the segmented images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_orig, cmap='gray')
    plt.title("Original Noisy Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image, cmap='gray')
    plt.title("Segmented Image (HMC)")
    plt.axis('off')
    
    res_path = join(reporesult, f'{image_stem}_segmentation_result.png')
    plt.savefig(res_path, bbox_inches='tight')
    print(f"Results saved to {res_path}")
    
    # Plot histogram and gaussian mixtures
    plt.figure()
    plt.hist(Y, bins=50, density=True, alpha=0.6, color='b', label='Image Histogram')
    
    x_axis = np.linspace(np.min(Y), np.max(Y), 1000)
    mixture = np.zeros_like(x_axis)
    final_I = ITabIter[nbIter-1, :]
    final_var = varTabIter[nbIter-1, :]
    for k in range(K):
        pdf_k = norm.pdf(x_axis, loc=final_means[k], scale=np.sqrt(final_var[k]))
        mixture += final_I[k] * pdf_k
        plt.plot(x_axis, final_I[k] * pdf_k, '--', label=f'Class {k}')
        
    plt.plot(x_axis, mixture, 'k-', linewidth=2, label='Mixture Model')
    plt.legend()
    plt.title("Histogram and Final Estimated Mixture")
    hist_path = join(reporesult, f'{image_stem}_segmentation_histogram.png')
    plt.savefig(hist_path, bbox_inches='tight')
    print(f"Histogram saved to {hist_path}")
    
    # Plot parameter convergence
    DrawCurvesParam(nbIter, join(reporesult, f'{image_stem}_segmentation'), meanTabIter, varTabIter, tTabIter)
    print("Parameter convergence plots saved.")
