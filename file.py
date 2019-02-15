#Face aligner
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner
import imutils
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('E:\Abhishek\OpenCV\Projects\shape_predictor_68_face_landmarks.dat')
face_align=FaceAligner(predictor)


image = cv2.imread('img3.jpg') #Replace it with your image.
image = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

dim = detector(gray, 2)
for i in dim:
    print('sd')
    (x,y,w,h)=rect_to_bb(i)
#    new_img=imutils.resize(image[y:y+h,x:x+w])
#    cv2.imshow('Face',new_img)
    aligned=face_align.align(image,gray,i)
    cv2.imshow('AlignedFace',aligned)
#    image[y:y+h,x:x+w]=aligned
#    cv2.imshow("Image after edit",image)






cv2.waitKey(0)
cv2.destroyAllWindows()
