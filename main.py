from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class PairMatch:
    i: int
    j: int
    homography_ij: np.ndarray
    inliers: int
    good_matches: int


def collect_image_paths(data_dir: Path) -> List[Path]:
    paths = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths)


def load_images(paths: List[Path]) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to load image: {p}")
            continue
        images.append(img)
    return images


def detect_features(images: List[np.ndarray]) -> Tuple[List[List[cv2.KeyPoint]], List[np.ndarray]]:
    orb = cv2.ORB_create(nfeatures=7000, scaleFactor=1.2, nlevels=8)
    keypoints_all: List[List[cv2.KeyPoint]] = []
    descriptors_all: List[np.ndarray] = []

    for idx, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, desc = orb.detectAndCompute(gray, None)
        if desc is None or len(kps) < 10:
            raise RuntimeError(f"Image index {idx} has too few features to stitch.")
        keypoints_all.append(kps)
        descriptors_all.append(desc)
    return keypoints_all, descriptors_all


def estimate_pair_homography(
    i: int,
    j: int,
    keypoints_all: List[List[cv2.KeyPoint]],
    descriptors_all: List[np.ndarray],
    ratio: float = 0.75,
) -> PairMatch | None:
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(descriptors_all[i], descriptors_all[j], k=2)

    good = []
    for m_n in knn:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 20:
        return None

    pts_i = np.float32([keypoints_all[i][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_j = np.float32([keypoints_all[j][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    h_ij, mask = cv2.findHomography(pts_i, pts_j, cv2.RANSAC, 4.0)
    if h_ij is None or mask is None:
        return None

    inliers = int(mask.ravel().sum())
    if inliers < 18:
        return None

    return PairMatch(i=i, j=j, homography_ij=h_ij, inliers=inliers, good_matches=len(good))


def build_pairwise_matches(
    keypoints_all: List[List[cv2.KeyPoint]],
    descriptors_all: List[np.ndarray],
) -> Tuple[Dict[Tuple[int, int], PairMatch], np.ndarray]:
    n = len(keypoints_all)
    pair_matches: Dict[Tuple[int, int], PairMatch] = {}
    score = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            pm = estimate_pair_homography(i, j, keypoints_all, descriptors_all)
            if pm is None:
                continue
            pair_matches[(i, j)] = pm
            score[i, j] = pm.inliers
            score[j, i] = pm.inliers

    return pair_matches, score


def build_maximum_spanning_tree(score: np.ndarray, root: int) -> Dict[int, int]:
    n = score.shape[0]
    parent: Dict[int, int] = {root: -1}
    in_tree = np.zeros(n, dtype=bool)
    in_tree[root] = True

    best_score = np.full(n, -1.0, dtype=np.float32)
    best_parent = np.full(n, -1, dtype=np.int32)

    for v in range(n):
        if v != root:
            best_score[v] = score[root, v]
            best_parent[v] = root

    for _ in range(n - 1):
        candidates = np.where(~in_tree)[0]
        if len(candidates) == 0:
            break
        next_node = candidates[np.argmax(best_score[candidates])]
        if best_score[next_node] <= 0:
            break

        in_tree[next_node] = True
        parent[next_node] = int(best_parent[next_node])

        for v in np.where(~in_tree)[0]:
            if score[next_node, v] > best_score[v]:
                best_score[v] = score[next_node, v]
                best_parent[v] = next_node

    return parent


def compute_global_transforms(
    n: int,
    parent: Dict[int, int],
    pair_matches: Dict[Tuple[int, int], PairMatch],
    root: int,
) -> Dict[int, np.ndarray]:
    transforms: Dict[int, np.ndarray] = {root: np.eye(3, dtype=np.float64)}
    remaining = set(parent.keys()) - {root}

    while remaining:
        progressed = False
        for node in list(remaining):
            p = parent[node]
            if p not in transforms:
                continue

            if (node, p) in pair_matches:
                h_node_to_parent = pair_matches[(node, p)].homography_ij
            elif (p, node) in pair_matches:
                h_parent_to_node = pair_matches[(p, node)].homography_ij
                h_node_to_parent = np.linalg.inv(h_parent_to_node)
            else:
                continue

            transforms[node] = transforms[p] @ h_node_to_parent
            remaining.remove(node)
            progressed = True

        if not progressed:
            break

    if len(transforms) < 2:
        raise RuntimeError("Could not compute global transforms; images may not overlap enough.")

    if len(transforms) < n:
        missing = sorted(set(range(n)) - set(transforms.keys()))
        print(f"[WARN] Some images are disconnected and will be skipped: {missing}")

    return transforms


def compute_canvas(images: List[np.ndarray], transforms: Dict[int, np.ndarray]) -> Tuple[np.ndarray, Tuple[int, int]]:
    corners_all = []
    for idx, img in enumerate(images):
        if idx not in transforms:
            continue
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, transforms[idx])
        corners_all.append(warped)

    all_pts = np.vstack(corners_all).reshape(-1, 2)
    min_xy = np.floor(all_pts.min(axis=0)).astype(np.int32)
    max_xy = np.ceil(all_pts.max(axis=0)).astype(np.int32)

    tx, ty = -min_xy[0], -min_xy[1]
    width = int(max_xy[0] - min_xy[0])
    height = int(max_xy[1] - min_xy[1])

    offset = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    return offset, (width, height)


def blend_images(images: List[np.ndarray], transforms: Dict[int, np.ndarray], offset: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    acc = np.zeros((height, width, 3), dtype=np.float32)
    wsum = np.zeros((height, width), dtype=np.float32)

    for idx, img in enumerate(images):
        if idx not in transforms:
            continue

        h, w = img.shape[:2]
        warp_mat = offset @ transforms[idx]

        warped_img = cv2.warpPerspective(img.astype(np.float32), warp_mat, (width, height))

        base_mask = np.full((h, w), 255, dtype=np.uint8)
        warped_mask = cv2.warpPerspective(base_mask, warp_mat, (width, height))
        valid = (warped_mask > 0).astype(np.uint8)

        # Distance-based weights reduce seams at overlap boundaries.
        dist = cv2.distanceTransform(valid, cv2.DIST_L2, 3)
        weight = np.where(valid > 0, np.maximum(dist, 1e-3), 0.0).astype(np.float32)

        acc += warped_img * weight[:, :, None]
        wsum += weight

    result = np.zeros_like(acc, dtype=np.uint8)
    valid = wsum > 0
    result[valid] = (acc[valid] / wsum[valid, None]).clip(0, 255).astype(np.uint8)
    return result


def stitch_all_images(data_dir: Path, output_path: Path) -> None:
    image_paths = collect_image_paths(data_dir)
    if not image_paths:
        print(f"[INFO] No images found under: {data_dir}")
        print("       Put input images in data/ and run again.")
        return

    print(f"[INFO] Found {len(image_paths)} image(s)")
    for p in image_paths:
        print(f"       - {p}")

    images = load_images(image_paths)
    if len(images) < 2:
        raise RuntimeError("Need at least 2 valid images to stitch.")

    print("[INFO] Detecting features...")
    kps, desc = detect_features(images)

    print("[INFO] Building pairwise matches and homographies...")
    pair_matches, score = build_pairwise_matches(kps, desc)
    if len(pair_matches) == 0:
        raise RuntimeError("No overlapping image pairs found.")

    root = int(np.argmax(score.sum(axis=1)))
    print(f"[INFO] Root image index selected: {root}")

    parent = build_maximum_spanning_tree(score, root)
    if len(parent) < 2:
        raise RuntimeError("Could not connect images into a valid matching graph.")

    transforms = compute_global_transforms(len(images), parent, pair_matches, root)

    offset, size = compute_canvas(images, transforms)
    print(f"[INFO] Panorama canvas size: {size[0]}x{size[1]}")

    print("[INFO] Warping and blending...")
    pano = blend_images(images, transforms, offset, size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), pano)
    if not ok:
        raise RuntimeError(f"Failed to save output image: {output_path}")

    print(f"[DONE] Panorama saved to: {output_path}")


def main() -> None:
    data_dir = Path("data")
    output_path = Path("output") / "panorama.jpg"
    stitch_all_images(data_dir=data_dir, output_path=output_path)


if __name__ == "__main__":
    main()
