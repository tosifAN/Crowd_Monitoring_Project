import numpy as np
import cv2
from ultralytics import YOLO


feature_params = dict(maxCorners=5,
                      qualityLevel=0.3,
                      minDistance=10,
                      blockSize=4)


trajectory_len = 20
detect_interval = 5
trajectories = []
frame1_idx = 0


cap = cv2.VideoCapture('Video_path')
model = YOLO('yolov8n-pose.pt')  
fgbg = cv2.createBackgroundSubtractorMOG2()
ok, frame2 = cap.read()
what, frame3 = cap.read()

parameters_shitomasi = dict(maxCorners=100, qualityLevel=0.3, minDistance=7)

frame1_gray_init = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

edges = cv2.goodFeaturesToTrack(
    frame1_gray_init, mask=None, **parameters_shitomasi)

canvas1 = np.zeros_like(frame2)
canvas2 = np.zeros_like(frame3)

colours = np.random.randint(0, 255, (100, 3))

parameter_lucas_kanade = dict(winSize=(15, 15), maxLevel=2, criteria=(
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))


def select_point(event, x, y, flags, params):
    global point, selected_point, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        selected_point = True
        old_points = np.array([[x, y]], dtype=np.float32)


cv2.namedWindow('Specific Object Tracking')
cv2.setMouseCallback('Specific Object Tracking', select_point)

selected_point = False
point = ()
old_points = ([[]])

frame1_gra_init = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)


while True:
    suc, frame1 = cap.read()
    ok, frame2 = cap.read()
    what, frame3 = cap.read()
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    img = frame1.copy()
    frame1_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    update_edges, status, errors = cv2.calcOpticalFlowPyrLK(
        frame1_gray_init, frame1_gray, edges, None, **parameter_lucas_kanade)
    new_edges = update_edges[status == 1]
    old_edges = edges[status == 1]

    # adding pose code here;
    if ok:
        results = model(frame2, save=True)
        annotated_frame1 = results[0].plot()
        cv2.imshow("Pose Tracking Using YoloV8", annotated_frame1)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

    # adding backgroud subtraction
    fgmask = fgbg.apply(frame2)

    cv2.imshow('Movement Tracking Using Background Subtraction', fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    for i, (new, old) in enumerate(zip(new_edges, old_edges)):
        a, b = new.ravel()
        c, d = old.ravel()
        mas = cv2.line(canvas1, (int(a), int(b)),
                       (int(c), int(d)), colours[i].tolist(), 2)
        frame2 = cv2.circle(frame1, (int(a), int(b)), 5, colours[i].tolist(), -1)

    result = cv2.add(frame2, mas)
    frame1_gray_init = frame1_gray.copy()
    edges = new_edges.reshape(-1, 1, 2)
    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame1_gray
        p0 = np.float32([trajectory[-1]
                        for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(
            img0, img1, p0, None, **parameter_lucas_kanade)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(
            img1, img0, p1, None, **parameter_lucas_kanade)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1

        new_trajectories = []

        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            trajectory.append((x, y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 50), -1)

        trajectories = new_trajectories

        cv2.polylines(img, [np.int32(trajectory)
                      for trajectory in trajectories], False, (0, 50, 0))
        cv2.putText(img, 'track count: %d' % len(trajectories),
                    (50, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 50, 0), 2)

    if frame1_idx % detect_interval == 0:
        mask = np.zeros_like(frame1_gray)
        mask[:] = 255

        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        p = cv2.goodFeaturesToTrack(frame1_gray, mask=mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])

    frame1_idx += 1
    prev_gray = frame1_gray

    frame1_gra = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

    if selected_point is True:
        cv2.circle(frame3, point, 5, (0, 0, 255), 2)
        new_points, status, errors = cv2.calcOpticalFlowPyrLK(frame1_gra_init, frame1_gra, old_points, None,
                                                              **parameter_lucas_kanade)
        frame1_gra_init = frame1_gra.copy()
        old_points = new_points

        x, y = new_points.ravel()
        j, k = old_points.ravel()

        canvas2 = cv2.line(canvas2, (int(x), int(y)),
                         (int(j), int(k)), (0, 255, 0), 3)
        frame3 = cv2.circle(frame3, (int(x), int(y)), 5, (0, 255, 0), -1)

    cv2.imshow('Count Object and Tracking(Main_window)', img)
    cv2.imshow('Count Object and Tracking(Mask_window)', mask)
    cv2.imshow('Object Detection and Path Tracking (Main_window)', frame2)
    cv2.imshow('Object Detection and Path Tracking (Mask_window)', mas)
    resul = cv2.add(frame3, canvas2)
    cv2.imshow('Specific Object Tracking(click on object)', resul)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
