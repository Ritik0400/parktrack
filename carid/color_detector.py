import cv2
import numpy as np
from sklearn.cluster import KMeans

# Semantic HSV ranges for mapping
COLOR_RANGES = {
    "red":      [(0, 40, 40), (10, 255, 255)],
    "red2":     [(170, 40, 40), (180, 255, 255)],
    "yellow":   [(15, 40, 40), (35, 255, 255)],
    "green":    [(35, 40, 40), (85, 255, 255)],
    "cyan":     [(85, 40, 40), (100, 255, 255)],
    "blue":     [(100, 40, 40), (130, 255, 255)],
    "purple":   [(130, 40, 40), (160, 255, 255)],
    "white":    [(0, 0, 200), (180, 40, 255)],
    "gray":     [(0, 0, 60), (180, 40, 200)],
    "black":    [(0, 0, 0), (180, 255, 60)],
}

def dominant_color_name(bgr_crop):
    """
    Returns (color_name, confidence)
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return "unknown", 0.0

    # Resize smaller for speed
    img = cv2.resize(bgr_crop, (80, 80), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.reshape((-1, 3))

    # Use KMeans to get dominant cluster
    try:
        kmeans = KMeans(n_clusters=3, n_init=5).fit(hsv)
        counts = np.bincount(kmeans.labels_)
        main_cluster = kmeans.cluster_centers_[np.argmax(counts)]
        h, s, v = main_cluster
    except:
        return "unknown", 0.0

    # Compare cluster centroid to semantic ranges
    for cname, (low, high) in COLOR_RANGES.items():
        low = np.array(low)
        high = np.array(high)
        if low[0] <= h <= high[0] and low[1] <= s <= high[1] and low[2] <= v <= high[2]:
            # Confidence = proportion of pixels in main cluster
            conf = counts.max() / counts.sum()
            return cname.replace("2", ""), float(conf)

    return "unknown", 0.0
