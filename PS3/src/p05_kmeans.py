import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os


# Number of clusters (ie number of colors)
N_CLUSTERS = 16
MIN_ITERS = 30
MAX_ITERS = 1000
TOL = 1e-5

def train_k_means(image, MIN_ITERS, MAX_ITERS, TOL):

    # Ramdomly initialize clusters centroids
    height, width, _ = image.shape
    idx = np.random.choice(height * width, N_CLUSTERS, replace=False)
    centroids = image[idx // width, idx % width].astype('float')
    
    # K-Means Algorithm
    it = 0
    prev_centroids = np.zeros((N_CLUSTERS, 3))
    print("---- Running K-Means algorithm ----")
    while it < MIN_ITERS or diff >= TOL:

        # Assign each pixel to its closest centroid
        c = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                c[i, j] = np.argmin(np.linalg.norm(image[i, j] - centroids, 2, axis=1)**2)

        # Move each cluster to mean of assigned points clusters
        for j in range(N_CLUSTERS):
            centroids[j] = np.mean(image[(c == j)], axis=0)

        # Finish iteration
        it += 1
        diff = np.linalg.norm(centroids - prev_centroids, 1)
        prev_centroids = centroids.copy()
        if it % 10 == 0:
            print(f"- Iteration {it} finished, centroids difference = {diff}")
        if it == MAX_ITERS:
            print("---- K-Means reached max iterations without convergeance ----")
            return centroids

    # Return centroids
    print(f"---- K-Means converged in {it} iterations ----")
    return centroids


def apply_clustering(image, centroids):

    # Assign each pixel to its closest centroid
    image_compressed = image.copy()
    height, width, _ = image.shape
    for i in range(height):
        for j in range(width):
            c_ij = np.argmin(np.linalg.norm(image[i, j] - centroids, 2, axis=1)**2)
            image_compressed[i, j] = centroids[c_ij]
    return image_compressed

def main():
    
    for size in ["small", "large"]:

        # Load image
        image_path = os.path.join('..', 'data', 'peppers-{}.tiff'.format(size))
        image = imread(image_path)
        
        # Plot original image
        plt.imshow(image)
        plt.savefig('output/p05_orig_{}.png'.format(size))

        # Get centroids for k means
        if size == "small":
            centroids = train_k_means(image, MIN_ITERS, MAX_ITERS, TOL)
        
        # Compress image by applying the clustering
        image_compressed = apply_clustering(image, centroids)

        # Plot compressed image
        plt.imshow(image_compressed)
        plt.savefig('output/p05_compr_{}.png'.format(size))
        print(f"---- The {size} image is compressed ---- ")

    return



if __name__ == "__main__":
    main()