import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    """
    Calculates the angle (in degrees) between three points a, b, c. Angle is at point b.
    Each point = (x, y). 
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))

    return angle


def extract_angles(landmarks, width, height):
    """
    Extracts important joint angles from MediaPipe landmarks.
    Returns a dictionary of angles.
    """
    # Convert normalized landmarks to pixel coords
    #intiially you get normalized coordinates:btwn 0 and 1, we convert them to pixel coordinates
    def lm(index):
        return [landmarks[index].x * width, landmarks[index].y * height]

    # Example joints, numbers based on MediaPipe's landmark indexing
    left_shoulder, left_elbow, left_wrist = lm(11), lm(13), lm(15)
    right_shoulder, right_elbow, right_wrist = lm(12), lm(14), lm(16)
    left_hip, left_knee, left_ankle = lm(23), lm(25), lm(27)
    right_hip, right_knee, right_ankle = lm(24), lm(26), lm(28)

    angles = {
        "left_elbow": calculate_angle(left_shoulder, left_elbow, left_wrist),
        "right_elbow": calculate_angle(right_shoulder, right_elbow, right_wrist),
        "left_knee": calculate_angle(left_hip, left_knee, left_ankle),
        "right_knee": calculate_angle(right_hip, right_knee, right_ankle),
    }
    return angles


def compare_angles(live_angles, ref_angles, threshold=15):
    """
    Compares live pose angles with reference angles.
    Returns feedback messages.
    """
    feedback = []

    for joint, ref_value in ref_angles.items():
        if joint in live_angles:
            diff = live_angles[joint] - ref_value
            if abs(diff) > threshold:
                if diff > 0:
                    feedback.append(f"{joint}: lower it")
                else:
                    feedback.append(f"{joint}: raise it")
    return feedback


# -------------------------------
# Main Program
# -------------------------------

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# 1. Load reference pose image
ref_path = input("Enter path of reference pose image: ")
ref_img = cv2.imread(ref_path)
if ref_img is None:
    print("Error: Could not load reference image.")
    exit()

ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) #convert to RGB, as mediapipe uses RGB format
ref_results = pose.process(ref_rgb)

if not ref_results.pose_landmarks:
    print("No pose detected in reference image.")
    exit()

ref_h, ref_w = ref_img.shape[:2]
ref_angles = extract_angles(ref_results.pose_landmarks.landmark, ref_w, ref_h)
print("Reference angles:", ref_angles)

# Draw reference skeleton
ref_skeleton = ref_img.copy()
mp_drawing.draw_landmarks(
    ref_skeleton,
    ref_results.pose_landmarks,
    mp_pose.POSE_CONNECTIONS,
    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
)

cv2.imshow("Reference Pose Skeleton", ref_skeleton)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Live webcam feed
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror view
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0,255,255), thickness=2)
        )
        # Extract live angles
        live_angles = extract_angles(results.pose_landmarks.landmark, frame.shape[1], frame.shape[0])

        # Compare with reference angles
        feedback = compare_angles(live_angles, ref_angles, threshold=15)

        # Display feedback on screen
        y0 = 30
        for i, msg in enumerate(feedback):
            cv2.putText(frame, msg, (10, y0 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # Show window
    cv2.imshow("Pose Correction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
