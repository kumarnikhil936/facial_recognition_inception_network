import glob
import os
from fr_utils import *
from model import *
import keras

K.set_image_data_format('channels_first')

pad = 50


def triplet_loss(y_true, y_pred, alpha=0.3):
    """
    Implementation of the triplet loss as defined by formula (3)
    Arguments:
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss


def prepare_database():
    database = {}

    cv2.namedWindow('Input image', cv2.WINDOW_NORMAL)

    # load all the images of individuals to recognize into the database
    # for file in glob.glob("./images1/*/*.jpg"):
    for file in glob.glob("./images1/*/*.jpg"):
        person_name = os.path.split(os.path.dirname(file))[1]
        print("Loading image for : ", person_name, "\n")

        frame = cv2.imread(file, 1)
        cv2.imshow('Input image', frame)
        cv2.waitKey(10)
        
        identity = os.path.splitext(os.path.basename(file))[0]
        database[file] = img_path_to_encoding(file, model)
        # database[identity] = img_path_to_encoding(file, model)
    
    cv2.destroyAllWindows()   

    return database


def face_recognizer(database):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    image_path = "./nkj.jpg"
    print("Loading the test image from : ", image_path)

    frame = cv2.imread(image_path, 1)

    img = frame

    cv2.namedWindow('image to test', cv2.WINDOW_NORMAL)
    cv2.imshow('image to test', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img = process_frame(img, frame, face_cascade, database)

    # cv2.namedWindow('output image', cv2.WINDOW_NORMAL)
    # cv2.imshow('output image', img)
    # cv2.waitKey(100)
    # cv2.destroyAllWindows()


def process_frame(img, frame, face_cascade, database):
    """
    Determine whether the current frame contains the faces of people from our database
    """

    grayimage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    cv2.namedWindow('gray image', cv2.WINDOW_NORMAL)
    cv2.imshow('gray image', grayimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    faces = face_cascade.detectMultiScale(grayimage, 1.3, 5)

    # Loop through all the faces detected and determine whether or not they are in the database
    identities = []
    
    for (x, y, w, h) in faces:
        x1 = x - pad
        y1 = y - pad
        x2 = x + w + pad
        y2 = y + h + pad

        # img = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 250, 0), thickness=4, shift=2)

        identity = find_identity(frame, x1, y1, x2, y2, database)

        if identity is not None:
            identities.append(identity)
            print("\n ****** The person(s) detected in the image is ", identity, "******\n \n")

    # if identities != []:
      #   cv2.imwrite('detected.png', img)

    return img


def find_identity(frame, x1, y1, x2, y2, database):
    """
    Determine whether the face contained within the bounding box exists in our database
    """
    # Shape of image is accessed by img.shape. It returns a tuple of number of rows, columns and channels.

    height, width, channels = frame.shape

    # The pad is necessary since the OpenCV face detector creates the bounding box around the face and not the head
    partial_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]

    return who_is_it(partial_image, database, model)


def who_is_it(image, database, model):
    """
    Implements face recognition by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
   
    print("The face boundary found in the image is :\n ")
    cv2.namedWindow('face_boundary', cv2.WINDOW_NORMAL)
    cv2.imshow('face boundary', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('detected_face.png', image)

    encoding = img_to_encoding(image, model)

    min_dist = 10
    identity = None

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)

        person_name = os.path.split(os.path.dirname(name))[1]
        print('Distance of encodings of test image from %s image is %s' % (person_name, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = person_name

    if min_dist > 0.78:
        return None
    else:
        return str(identity)


if __name__ == "__main__":
    model = facenetModel(input_shape=(3, 96, 96))
    # model.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
    load_weight(model)
    model.save('model.h5')

    # model = keras.models.load_model('model.h5')

    # keras.utils.print_summary(model)

    database = prepare_database()

    face_recognizer(database)
