import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
from numpy import random
from tkinter import filedialog
from matplotlib import pyplot as plt
import customtkinter as ctk

# Global variable for storing the file path of the selected image or video
file_path = ''

def select_input():
    # Open a file dialog to select the input image or video
    global file_path
    file_path = tk.filedialog.askopenfilename()
    # Load the selected image or video
    input_image = cv2.imread(file_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(input_image)
    input_image = ctk.CTkImage(input_image,size=(200,200))
    # Update the input label with the image with the detected faces
    input_label.configure(image=input_image)
    input_label.image = input_image
    
  

def detect_faces(file_path):
    # Load the input image or video
    input_image = cv2.imread(file_path)
    
    # Load the face detection classifier from the OpenCV library
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(input_image, 1.3, 5)
    # Draw a rectangle around the faces
    for (x,y,w,h) in faces:
        cv2.rectangle(input_image, (x,y), (x+w,y+h), (255,0,0), 2)
    # display the image
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(input_image)
    input_image = ctk.CTkImage(input_image,size=(200,200))
    # Update the input label with the image with the detected faces
    detect_label.configure(image=input_image)
    detect_label.image = input_image


def blur_faces(file_path,value):
    global finalimg
    # Load the input image or video
    input_image = cv2.imread(file_path)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # Load the face detection classifier from the OpenCV library
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
    # Blur the detected faces
    for (x,y,w,h) in faces:
        face_region = input_image[y:y+h, x:x+w]
        face_region = cv2.GaussianBlur(face_region, (15,15),value)
        input_image[y:y+h, x:x+w] = face_region
    # Convert the image from BGR to RGB

    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
     
    finalimg = input_image
    # Convert the image to a PhotoImage object
    input_image = Image.fromarray(input_image)
    input_image = ctk.CTkImage(input_image,size=(200,200))
    # Update the input label with the image with the detected faces blurred
    blur_label.configure(image=input_image)
    blur_label.image = input_image

def unblur_faces():
    # Load the input image or video
    input_image = finalimg
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(input_image, 1.3, 5)
    # Unblur the detected faces
    for (x,y,w,h) in faces:
        face_region = input_image[y:y+h, x:x+w]
        face_region = cv2.GaussianBlur(face_region, (15,15), -20)
        input_image[y:y+h, x:x+w] = face_region
    # Convert the image from BGR to RGB
    #input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    # Convert the image to a PhotoImage object
    input_image = Image.fromarray(input_image)
    input_image = ctk.CTkImage(input_image,size=(200,200))
    # Update the input label with the image with the detected faces unblurred
    unblur_label.configure(image=input_image)
    unblur_label.image = input_image
    

    
def addnoise(file_path,value):
    global denoise
    value = int(value)
    # Read the input image using OpenCV
    input_image = cv2.imread(file_path)
    noise_image = input_image
    # Get the shape of the image (number of rows, number of columns, and number of channels)
    r, c, _ = noise_image.shape
    # Generate a random number of pixels to change
    pixl = random.randint(2000, 5000)
    
    # Change pixl random pixels to 255 (white)
    for i in range(pixl):
        y = random.randint(0, r - 1)
        x = random.randint(0, c - 1)
        noise_image[y][x] = 255
    
    # Change pixl random pixels to 0 (black)
    for i in range(pixl):
        y = random.randint(0, r - 1)
        x = random.randint(0, c - 1)
        noise_image[y][x] = 0
    denoise = noise_image
    final_noise_image = cv2.cvtColor(noise_image, cv2.COLOR_BGR2RGB)
    final_noise_image = Image.fromarray(final_noise_image)
    final_noise_image = ctk.CTkImage(final_noise_image,size=(200,200))
    # Update the input label with the image with the detected faces
    noise_label.configure(image=final_noise_image)
    noise_label.image = final_noise_image
    

def remove_noise_from_noise():
    # Read the input image using OpenCV
    input_image = denoise
    denoise_image_from_noise = cv2.medianBlur(input_image,5)
    final_denoise_image = cv2.cvtColor(denoise_image_from_noise, cv2.COLOR_BGR2RGB)
    final_denoise_image = Image.fromarray(final_denoise_image)
    final_denoise_image = ctk.CTkImage(final_denoise_image,size=(200,200))
    # Update the input label with the image with the detected faces
    denoise_label.configure(image=final_denoise_image)
    denoise_label.image = final_denoise_image
def create_widgets():
    global blur_scale,noise_scale
    blur_scale = ctk.CTkSlider(root, from_=0, to=100, number_of_steps=5, command=lambda value: blur_faces(file_path,float(value)))
    blur_scale.grid(row=5, column=6)
    noise_scale = ctk.CTkSlider(root, from_=0, to=100,number_of_steps=5, command=lambda value: addnoise(file_path,float(value)))
    noise_scale.grid(row=6, column=6)

root = ctk.CTk()
root.title("Computer Vision_project")
root.geometry("400x400")
ctk.set_appearance_mode("dark")

title_label = ctk.CTkLabel(root, text="FACE OBFUSCATION TOOL", font=("Verdana",60), text_color=("white"))
title_label.grid(row=0, column=8, pady=10, padx=10, sticky="n")


# Create a button for selecting the input image or video
input_button = ctk.CTkButton(root, text='Select Input', command=select_input)
input_button.grid(row=4, column=4)

# Create a label for displaying the input image or video
input_label = ctk.CTkLabel(root,text=" ")
input_label.grid(row=4, column=5)

# Create a button for detecting faces in the input image or video
detect_button = ctk.CTkButton(root, text='Detect Faces', command=lambda: detect_faces(file_path))
detect_button.grid(row=4, column=7)
# Create a label for displaying the image with the detected faces
detect_label = ctk.CTkLabel(root,text=" ")
detect_label.grid(row=4, column=8)

# Create a button for blurring the detected faces in the input image or video
blur_button = ctk.CTkButton(root, text='Blur Faces', command=lambda: blur_faces(file_path,blur_scale.get()))
blur_button.grid(row=5, column=4)
# Create a label for displaying the image with the detected faces blurred
blur_label = ctk.CTkLabel(root,text=" ")
blur_label.grid(row=5, column=5)

# Create a button for unblurring the detected faces in the input image or video
unblur_button_2 = ctk.CTkButton(root, text='Unblur from blur', command=lambda: unblur_faces())
unblur_button_2.grid(row=5, column=7)
# Create a label for displaying the input image or video
unblur_label = ctk.CTkLabel(root,text=" ")
unblur_label.grid(row=5, column=8)

# Create a button for adding noise to the input image or video
noise_button = ctk.CTkButton(root, text='Noise', command=lambda: addnoise(file_path,noise_scale.get()))
noise_button.grid(row=6, column=4)

noise_label = ctk.CTkLabel(root,text=" ")
noise_label.grid(row=6, column=5)

denoised_button_from_noise=ctk.CTkButton(root,text='Denoise_from_noise',command=lambda:remove_noise_from_noise())
denoised_button_from_noise.grid(row=6,column=7)

denoise_label = ctk.CTkLabel(root,text=" ")
denoise_label.grid(row=6, column=8)
create_widgets()
ctk.deactivate_automatic_dpi_awareness()

# Run the main loop
root.mainloop()
