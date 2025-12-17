import cv2
import numpy as np
import time
import json

# Define board parameters
SQUARES_X = 8
SQUARES_Y = 5
SQUARE_SIZE = 0.025  # meters
MARKER_SIZE = 0.015  # meters

# Load ArUco dictionary
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Create ChArUco board
board = cv2.aruco.CharucoBoard(
    size=(SQUARES_X, SQUARES_Y),
    squareLength=SQUARE_SIZE,
    markerLength=MARKER_SIZE,
    dictionary=ARUCO_DICT
)

# Initialize calibration data
all_charuco_corners = []
all_charuco_ids = []

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ ERROR: Could not open camera")

print("📷 Move the ChArUco board to collect calibration frames...")

frames_collected = 0
required_frames = 20
last_capture_time = time.time()
capture_interval = 1  # seconds

# FIXED: Use correct detector parameters
detector = cv2.aruco.ArucoDetector(ARUCO_DICT)

while frames_collected < required_frames:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame capture failed")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect markers (updated detection method)
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is not None and len(ids) > 0:
        # FIXED: Use updated interpolation function
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners,
            ids,
            gray,
            board
        )

        if retval > 10:  # Minimum corners required
            current_time = time.time()
            if current_time - last_capture_time > capture_interval:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                frames_collected += 1
                last_capture_time = current_time
                print(f"✅ Frame {frames_collected}/{required_frames} captured")
            
            # Draw detected corners
            if charuco_corners is not None and charuco_ids is not None:
                cv2.aruco.drawDetectedCornersCharuco(
                    frame, 
                    charuco_corners, 
                    charuco_ids
                )

    # Display info text
    cv2.putText(frame, f"Frames: {frames_collected}/{required_frames}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("ChArUco Calibration (Press Q to exit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Calibration
if len(all_charuco_corners) >= 10:
    print("\n📐 Calibrating camera...")
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=gray.shape[::-1],
        cameraMatrix=None,
        distCoeffs=None,
        flags=cv2.CALIB_FIX_ASPECT_RATIO
    )

    print(f"\n✅ Calibration successful | Reprojection error: {retval:.4f} px")
    print("📷 Camera Matrix:\n", camera_matrix)
    print("🔧 Distortion Coefficients:\n", dist_coeffs.ravel())

    # Save calibration data
    calib_data = {
        "image_size": gray.shape[::-1],
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.tolist(),
        "reprojection_error": retval
    }

    with open("charuco_calibration.json", "w") as f:
        json.dump(calib_data, f, indent=4)

    print("💾 Calibration data saved to 'charuco_calibration.json'")
else:
    print("❗ Insufficient frames for calibration (min 10 required)")
