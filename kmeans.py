import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

image_path = "image_path.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.flip(image, 0)

_, binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

white_points = np.column_stack(np.where(binary_image == 255))

n_clusters = 13

db = KMeans(n_clusters=n_clusters, random_state=10).fit(white_points)
labels = db.labels_

plt.figure(figsize=(15, 15))
plt.imshow(image, cmap='gray')

unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        continue

    class_member_mask = (labels == k)
    xy = white_points[class_member_mask]
    plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

    cluster_center = np.mean(xy, axis=0)
    cluster_radius = np.linalg.norm(xy - cluster_center, axis=1).max()
    circle = plt.Circle((cluster_center[1], cluster_center[0]), cluster_radius, color=tuple(col), fill=False, linewidth=2)
    plt.gca().add_patch(circle)

plt.title(f'Clusters con KMeans: {n_clusters}')
plt.gca().invert_yaxis()
plt.show()
