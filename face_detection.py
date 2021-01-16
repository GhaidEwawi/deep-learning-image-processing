!pip install face_recognition

import face_recognition
import PIL.Image
import PIL.ImageDraw

# Storing the image in a variable
image = face_recognition.load_image_file('people.jpg')

# Finding face locations in the image (pretrained HOG model)
face_locations = face_recognition.face_locations(image)
face_landmarks_list = face_recognition.face_landmarks(image)

number_of_faces = len(face_locations)
print(f"I found {number_of_faces} face(s) in this photograph")

# Load the image into a python Image Library object so that we can draw on top of it and display it
# this line only copies the images into the structure of pil_image.. because pill uses its own format.
# We did this because we'll use pil library for drawing a box around the detected face.
pil_image = PIL.Image.fromarray(image)

for face_location in face_locations:

    # Print the location of each face in this image, Each face is a list of co-ordinates in (top, right, bottom, left) order.
    top, right, bottom, left = face_location
    print(f"A face is located at pixel location Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}")

    # Let's draw a box around the face
    draw = PIL.ImageDraw.Draw(pil_image)
    draw.rectangle([left, top, right, bottom], outline="red")

for face_landmarks in face_landmarks_list:

    # Loop over each facial feature (eye, nose, mouth, lips, etc)
    for name, list_of_points in face_landmarks.items():

        # Print the location of each facial feature in this image
        print(f"The {name} in this face has the following points: {list_of_points}")

        # Let's trace out each facial feature in the image with a line!
        draw.line(list_of_points, fill='red', width=2)

# Display the image on screen
pil_image.show() # this didn't work so I used the code block below

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

plt.imshow(pil_image)
plt.show()