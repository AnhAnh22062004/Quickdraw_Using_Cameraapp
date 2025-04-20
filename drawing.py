from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import Image, ImageDraw, ImageTk
import cv2

class MyvideoCapture: 
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) # Get width of the frame
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) # Get height of the frame

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release() # Release the video capture object

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret: # Return a boolean value indicating if the frame was read correctly
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (None, None)

class Paint(object):
    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'blue'

    def __init__(self):
        self.root = Tk()
        self.root.title("Paint")
        self.root.vid = MyvideoCapture(0)
        self.width = int(self.vid.width) + 100
        self.height = int(self.vid.height)
        self.root.geometry(f"{self.width}x{self.height}")
        self.root.maxsize(self.width, self.height)
        self.root.minsize(self.width, self.height)

        self.paint_tools = Frame(self.root, width= self.width, height= self.height, relief= RIDGE, borderwidth=2)
        self.paint_tools.place(x=0, y=0)

        self.b = Label(self.paint_tools, borderwidth= 0, text= 'brush', font= ('Arial', 12, 'bold'))
        self.b.place(x=10, y=10)
        self.brush_button = Button(self.paint_tools, text= 'brush', command= self.brush)
        self.brush_button.place(x=60, y=40)

        self.cl = Label(self.paint_tools, borderwidth= 0, text= 'color', font= ('Arial', 12, 'bold'))
        self.cl.place(x=10, y=80)
        self.color_button = Button(self.paint_tools, text= 'color', command= self.chosses_color)
        self.color_button.place(x=60, y=110)
        
        self.e = Label(self.paint_tools, borderwidth= 0, text= 'eraser', font= ('Arial', 12, 'bold'))
        self.e.place(x=10, y=150)
        self.eraser_button = Button(self.paint_tools, text= 'eraser', command= self.use_eraser)
        self.eraser_button.place(x=60, y=180)

        self.pen_size = Label(self.paint_tools, borderwidth= 0, text= 'pen size', font= ('Arial', 12, 'bold'))
        self.pen_size.place(x=10, y=220)
        self.choose_size_button = Scale(self.paint_tools, from_=1, to=10, orient=VERTICAL)
        self.choose_size_button.set(5)
        self.choose_size_button.place(x=60, y=250)

        self.c = Canvas(self.root, bg= 'white', width= 1920, height= 1080, cursor= 'cross', borderwidth= 0, relief= RIDGE)
        self.c.place(x=100, y=0)

        self.setup()
        self.root.mainloop()

def setup(self): # Initialize the canvas and set up the drawing environment
    self.old_x = None
    self.old_y = None
    self.line_width = self.choose_size_button.get()
    self.color = self.DEFAULT_COLOR
    self.eraser_on = False
    self.active_button = self.brush_button

    self.delay = 15
    self.update()
    self.c.bind("<B1-Motion>", self.paint)
    self.c.bind("<ButtonRelease-1>", self.reset)

def use_bursh(self): # Set the brush tool as active
    self.active_button = self.brush_button

def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

def use_eraser(self):
    self.activate_button(self.eraser_button, eraser_mode=True)

def update(self):
    ret, frame = self.vid.get_frame()
    if ret:
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        bg = self.c.create_image(0, 0, image=self.photo, anchor=NW)
        self.c.tag_lower(bg)
        
    self.root.after(self.delay, self.update) # Call the update function after a delay

def paint(self, event): # Paint on the canvas when the mouse is moved with the left button pressed
    self.line_width = self.choose_size_button.get() 
    paint_color = self.color if not self.eraser_on else self.color 
    if self.old_x and self.old_y:
        self.c.create_line((self.old_x, self.old_y, event.x, event.y), 
                           width=self.line_width, fill=paint_color, capstyle=ROUND, smooth=True, splinesteps=36)
    self.old_x = event.x 
    self.old_y = event.y

def reset(self, event):
    self.old_x, self.old_y = None, None # Reset the old x and y coordinates to None when the mouse button is released

if __name__ == "__main__":
    Paint()
# This code creates a simple drawing application using Tkinter and OpenCV. The application allows the user to draw on a canvas using a brush tool, change the color of the brush, and use an eraser tool. The canvas is updated with frames from the webcam feed.`