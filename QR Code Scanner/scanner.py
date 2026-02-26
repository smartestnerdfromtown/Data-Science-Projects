import cv2

image = cv2.imread(filename="qr_codes/qr_code.png")

detector = cv2.QRCodeDetector()

data, bbox, _ = detector.detectAndDecode(image)

if bbox is not None:
    print("QR Code detected!")
    print("Decoded data:", data)

    bbox = bbox.astype(int)
    for i in range(len(bbox[0])):
        point1 = tuple(bbox[0][i])
        point2 = tuple(bbox[0][(i + 1) % len(bbox[0])])
        cv2.line(image, point1, point2, (0, 255, 0), 2)

    cv2.imshow("QR Code", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No QR Code found.")