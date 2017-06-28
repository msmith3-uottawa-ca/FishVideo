import cv2
cap = cv2.VideoCapture('S:\\Mark\\Research\\Fish Behavioural\\27062017 Bernard Looming\\Run1\\2017-06-27_11-15-56.mp4')


while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()