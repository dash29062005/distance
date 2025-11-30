import cv2
import numpy as np
from app.registration.manual_register import ManualRegister

# Create a simple test image (so test works even without real images)
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(img, "Draw ROI here", (150, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

# Convert to RGB (our system standard)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

reg = ManualRegister()

print(">>> Draw a box with the mouse and press ENTER")
print(">>> Press ESC to cancel")

crop, bbox = reg.select_roi(img_rgb)

print("\nRESULT:")

if crop is None:
    print("No ROI selected (cancel or zero box).")
else:
    print("BBOX:", bbox)
    print("Crop shape:", crop.shape)
    cv2.imshow("Selected ROI", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
