import cv2
import glob
import numpy as np

pattern_size = (11, 8)
samples = []

images = glob.glob('tmp/*.JPG')

for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, (800, 600))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res, corners = cv2.findChessboardCorners(img, pattern_size, None)

    img_show = img.copy()
    cv2.drawChessboardCorners(img_show, pattern_size, corners, res)
    cv2.putText(img_show, 'Muestra numero: %d' % (len(samples)+1), (0, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow('chessboard', img_show)

    wait_time = 0 if res else 30
    k = cv2.waitKey(wait_time)

    if k == ord('s') and res:
        samples.append((gray, corners))
    elif k == 27:
        break
cv2.destroyAllWindows()

criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 1e-3)

for i in range(len(samples)):
    img, corners =samples[i]
    corners = cv2.cornerSubPix(img, corners, (10, 10), (-1, -1), criteria)

pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

images, corners = zip(*samples)

pattern_points = [pattern_points]*len(corners)

rms, camera_matrix, dist_coefs, rvecs, tvecs =\
    cv2.calibrateCamera(pattern_points, corners, images[0].shape, None, None)

np.save('camera_mat.npy', camera_matrix)
np.save('dist_coefs.npy', dist_coefs)