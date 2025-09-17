import cv2 as cv
import numpy as np
from collections import deque

def make_K(w, h, f_factor=0.9):
    f = f_factor * max(w, h)
    return np.array([[f, 0, w/2],
                     [0, f, h/2],
                     [0, 0, 1]], dtype=np.float64)

def draw_trajectory(canvas, traj_pts, color=(0,0,0)):
    canvas[:] = 255
    if len(traj_pts) < 2: return canvas
    # Plot x-z (maju-mundur di z, kiri-kanan di x)
    pts = np.array([(int(300 + p[0]*10), int(550 - p[2]*10)) for p in traj_pts], dtype=np.int32)
    cv.polylines(canvas, [pts], isClosed=False, color=color, thickness=2)
    cv.circle(canvas, pts[-1], 3, (0,0,255), -1)
    return canvas

def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Tidak bisa akses webcam."); return

    ret, frame = cap.read()
    if not ret: return
    h, w = frame.shape[:2]
    K = make_K(w, h, f_factor=0.9)
    Kinv = np.linalg.inv(K)

    orb = cv.ORB_create(2000)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

    # Pose dunia (homogen 4x4)
    T_w_c = np.eye(4, dtype=np.float64)
    traj = deque(maxlen=3000)
    traj.append(np.array([0.0,0.0,0.0]))

    canvas = np.full((600,600,3), 255, np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        kp, des = orb.detectAndCompute(gray, None)
        if prev_des is None or des is None or len(kp) < 8 or len(prev_kp) < 8:
            cv.imshow("frame", frame)
            if cv.waitKey(1) == 27: break
            prev_gray, prev_kp, prev_des = gray, kp, des
            continue

        matches = bf.match(prev_des, des)
        if len(matches) < 100:
            # terlalu sedikit, lanjut saja
            prev_gray, prev_kp, prev_des = gray, kp, des
            cv.imshow("frame", frame)
            if cv.waitKey(1) == 27: break
            continue

        matches = sorted(matches, key=lambda m: m.distance)[:500]

        pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

        # Normalisasi dengan K
        pts1n = cv.undistortPoints(pts1.reshape(-1,1,2), K, None).reshape(-1,2)
        pts2n = cv.undistortPoints(pts2.reshape(-1,1,2), K, None).reshape(-1,2)

        E, inliers = cv.findEssentialMat(pts2, pts1, K, method=cv.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            prev_gray, prev_kp, prev_des = gray, kp, des
            cv.imshow("frame", frame)
            if cv.waitKey(1) == 27: break
            continue

        inliers = inliers.ravel().astype(bool)
        n_inl = inliers.sum()

        _, R, t, _ = cv.recoverPose(E, pts2[inliers], pts1[inliers], K)

        # Buat transform dari frame k-1 ke k (konvensi recoverPose)
        T_1to2 = np.eye(4); T_1to2[:3,:3] = R; T_1to2[:3,3] = t.ravel()  # skala 1.0

        # Akumulasi pose dunia
        T_w_c = T_w_c @ T_1to2
        cam_pos = T_w_c[:3, 3]
        traj.append(cam_pos.copy())

        # Visual
        vis = frame.copy()
        cv.putText(vis, f"ORB-VO | matches: {len(matches)} inliers: {n_inl}", (10,30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        for m in matches[:200]:
            p1 = tuple(np.int32(prev_kp[m.queryIdx].pt))
            p2 = tuple(np.int32(kp[m.trainIdx].pt))
            cv.line(vis, p1, p2, (0,255,0), 1)

        draw_trajectory(canvas, list(traj))
        cv.imshow("trajectory", canvas)
        cv.imshow("frame", vis)

        key = cv.waitKey(1)
        if key == 27: break  # ESC

        prev_gray, prev_kp, prev_des = gray, kp, des

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
