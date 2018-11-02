import cv2
import os
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=200)


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def main():
    create_folder('database/')

    while True:

        name = input("Enter Name: ")
        face_id = input("Enter id for this face: ")

        face_folder = 'database/' + str(face_id) + "/"
        create_folder(face_folder)
        break


    while True:
        init_img_no = input("Starting img no. (should be a int): ")
        init_img_no = int(init_img_no)
        break

    img_no = init_img_no

    # Capture a picture from the webcam
    cap = cv2.VideoCapture(0)

    # 10 images for each person
    total_images = 10

    while True:
        ret, img = cap.read()

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = detector(img_gray)

        if len(faces) == 1:
            face = faces[0]
            (x, y, w, h) = face_utils.rect_to_bb(face)

            face_img = img_gray[y - 50:y + h + 100, x - 50:x + w + 100]
            face_aligned = face_aligner.align(img, face_img, face)
            # face_aligned = face_aligner.align(img, img_gray, face)
            face_img = face_aligned

            img_path = face_folder + name + str(img_no) + ".jpg"
            cv2.imwrite(img_path, face_img)

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 3)
            cv2.imshow("aligned", face_img)

            img_no += 1

        cv2.imshow("Saving", img)

        cv2.waitKey(1)

        if img_no == init_img_no + total_images:
            break

    cap.release()


if __name__ == '__main__':
    main()
