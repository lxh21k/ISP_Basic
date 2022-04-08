import cv2

if __name__ == '__main__':
    img_path = "./IMG_2758.JPG"
    noisy_img = cv2.imread(img_path)

    result = cv2.bilateralFilter(noisy_img, 80, 75, 75)

    cv2.imshow("original image", noisy_img)
    cv2.imshow("result", result)
    cv2.waitKey()