# Some functions in this file are based on TkCurve by Akascape (https://github.com/Akascape/TkCurve)
# Licensed under the MIT License.
# See the original LICENSE at https://github.com/Akascape/TkCurve/blob/main/LICENSE

import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk, ImageDraw
import numpy as np 

import cv2
import h5py
import os
from pathlib import Path
from shapelysmooth import chaikin_smooth

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

class CurveWidget(tk.Canvas):
    """
    Curve line widget for tkinter
    Author: Akascape
    """
    def __init__(self,
                 parent,
                 points=[],
                 width=344,
                 height=464,
                 point_color="black",
                 point_size=8,
                 line_width=5,
                 line_color="orange",
                 outline="white",
                 grid_color="grey20",
                 bg="grey12",
                 smooth=True,
                 path = '',
                 **kwargs):
        
        super().__init__(parent, width=width, height=height, bg=bg, borderwidth=0, highlightthickness=0, **kwargs)
        self.width = width
        self.height = height
        self.line_color = line_color
        self.point_size = point_size
        self.line_width = line_width
        self.point_color = point_color
        self.outline_color = outline
        self.grid_color = grid_color
        self.smooth = smooth
        self.points = points
        self.path = path
        self.point_ids = []
        self.open_Image()
        self.create_curve()
        self.bind_events()
        self.x = tk.DoubleVar()
        self.y = tk.DoubleVar()
        self.bind("<Button-3>", self.get_mouse_position)

    def open_Image(self):
        global image
        if self.path:
            image = Image.open(self.path)
            image = image.resize(( self.width, self.height), Image.LANCZOS)
            image = ImageTk.PhotoImage(image)
            self.create_image(0, 0, image=image, anchor='nw') 
    def create_curve(self):
        if self.points==[]:
            self.points.append((round(self.width*0.5),round(self.height*0.95)))
       
        if len(self.points)==1:
            self.points.append((round(self.width*0.55),round(self.height*0.95)))
        
        self.create_line(self.points, tag='curve', fill=self.line_color, smooth=self.smooth, width=self.line_width,
                         capstyle=tk.ROUND, joinstyle=tk.BEVEL)

        for point in self.points:
            point_id = self.create_oval(point[0]-self.point_size, point[1]-self.point_size,
                                        point[0]+self.point_size, point[1]+self.point_size,
                                        fill=self.point_color, outline=self.outline_color, tags='point')
            self.point_ids.append(point_id)
            
    def bind_events(self):
        for point_id in self.point_ids:
            self.tag_bind(point_id, '<ButtonPress-1>', self.on_point_press)
            self.tag_bind(point_id, '<ButtonRelease-1>', self.on_point_release)
            self.tag_bind(point_id, '<B1-Motion>', self.on_point_move)

    def on_point_press(self, event):
        self.drag_data = {'x': event.x, 'y': event.y}

    def on_point_release(self, event):
        self.drag_data = {}
        current_id = self.find_withtag('current')[0]
        index = self.point_ids.index(current_id)
        
        if self.points[index][0]>self.winfo_width():
            dx = self.winfo_width() - self.points[index][0] - 8
            dy = 0
            self.move(current_id, dx, dy)
           
        if self.points[index][1]>self.winfo_height():
            dx = 0
            dy = self.winfo_height() - self.points[index][1] - 8
            self.move(current_id, dx, dy)

        if self.points[index][0]<0:
            dx = -self.points[index][0] + 8
            dy = 0
            self.move(current_id, dx, dy)
           
        if self.points[index][1]<0:
            dx = 0
            dy = -self.points[index][1] + 8
            self.move(current_id, dx, dy)

    def on_point_move(self, event):
        dx = event.x - self.drag_data['x']
        dy = event.y - self.drag_data['y']
        self.drag_data['x'] = event.x
        self.drag_data['y'] = event.y
        current_id = self.find_withtag('current')[0]
        self.move(current_id, dx, dy)
        index = self.point_ids.index(current_id)
        self.points[index] = (event.x, event.y)
        if len(self.points)==1:
            self.coords('curve', self.points[0][0], self.points[0][1],
                        self.points[0][0],self.points[0][1])
        else:
            self.coords('curve', sum(self.points, ()))
    
    def get_mouse_position(self, event):
        point = (event.x, event.y)
        self.add_point(point)
            
    def fix(self, point):
        if point in self.points:
            index = self.points.index(point)
            point_id = self.point_ids[index]
            self.tag_unbind(point_id, '<ButtonPress-1>')
            self.tag_unbind(point_id, '<ButtonRelease-1>')
            self.tag_unbind(point_id, '<B1-Motion>')

    def get(self):
        return self.points
    
    def add_point(self, point):
        if point in self.points:
            return
        self.points.append(point)
        point_id = self.create_oval(point[0]-self.point_size, point[1]-self.point_size,
                                        point[0]+self.point_size, point[1]+self.point_size,
                                        fill=self.point_color, outline=self.outline_color, tags='point')
        self.point_ids.append(point_id)
        self.tag_bind(point_id, '<ButtonPress-1>', self.on_point_press)
        self.tag_bind(point_id, '<ButtonRelease-1>', self.on_point_release)
        self.tag_bind(point_id, '<B1-Motion>', self.on_point_move)
        self.coords('curve', sum(self.points, ()))
        
    def delete_point(self, point):
        if point not in self.points:
            return
    
        point_id = self.point_ids[self.points.index(point)]
        self.points.remove(point)
        self.delete(point_id)
        if len(self.points)<=0:
            return
        if len(self.points)==1:
            self.coords('curve', self.points[0][0], self.points[0][1],
                        self.points[0][0],self.points[0][1])
        else:
            self.coords('curve', sum(self.points, ()))
       
    def config(self, **kwargs):
        if "point_color" in kwargs:
            self.point_color = kwargs.pop("point_color")
        if "outline" in kwargs:
            self.outline_color = kwargs.pop("outline")
        if "line_color" in kwargs:
            self.line_color = kwargs.pop("line_color")
        if "grid_color" in kwargs:
            self.grid_color = kwargs.pop("grid_color")
            self.itemconfig('grid_line', fill=self.grid_color)
        if "smooth" in kwargs:
            self.smooth = kwargs.pop("smooth")
        if "point_size" in kwargs:
            self.point_size = kwargs.pop("point_size")
        if "line_width" in kwargs:
            self.line_width = kwargs.pop("line_width")
        if "points" in kwargs:
            self.points = kwargs.pop("points")
            for i in self.point_ids:
                self.delete(i)
            self.point_ids = []
            
            for point in self.points:
                point_id = self.create_oval(point[0]-self.point_size, point[1]-self.point_size,
                                        point[0]+self.point_size, point[1]+self.point_size,
                                        fill=self.point_color, outline=self.outline_color, tags='point')
                self.point_ids.append(point_id)
            self.bind_events()
            
        for point_id in self.point_ids:
            self.itemconfig(point_id, fill=self.point_color, outline=self.outline_color)
            point = self.points[self.point_ids.index(point_id)]
            self.coords(point_id, point[0]-self.point_size, point[1]-self.point_size,
                        point[0]+self.point_size, point[1]+self.point_size)
            
        self.itemconfig('curve', fill=self.line_color, smooth=self.smooth, width=self.line_width)
        self.coords('curve', sum(self.points, ()))
        
        super().config(**kwargs)

    def cget(self, param):
        if param=="point_color":
            return self.point_color
        if param=="outline":
            return self.outline_color
        if param=="line_color":
            return self.line_color
        if param=="grid_color":
            return self.grid_color
        if param=="smooth":
            return self.smooth
        if param=="point_size":
            return self.point_size
        if param=="line_width":
            return self.line_width
        if param=="points":
            return self.points
        return super().cget(param)
    
    def delete_el(self):
        for item in self.point_ids:
            self.delete(item)
    def __del__(self):
        print("Curve_widget is deleted")

class MaskDrawerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw Mask")
        self.root.config(bg="black")
        self.root.geometry("900x600")

        self.selected_format = "hdf5"
        self.format_choice = tk.StringVar(value=self.selected_format)
        formats = ["hdf5", "png", "tiff", "bmp"]
        self.format_menu = tk.OptionMenu(self.root, self.format_choice, *formats, command=self.on_format_change)
        self.format_menu.grid(row=5, columnspan=2)

        self.save_path = tk.StringVar(value="no")

        self.mask = None
        self.canvas = None
        self.curve_widget = None

        self.path_list = []
        self.path_index = tk.IntVar(value=0)

        self.load_image_paths()
        if self.path_list:
            self.build_interface()
        else:
            tk.Label(root, text="No suitable files found.").pack()

    def load_image_paths(self):
        folder = filedialog.askdirectory(title="Select directory with moved_530.tiff files")
        if not folder:
            print("No folder selected.")
            return

        self.folder = Path(folder)
        seen = set()

        for file in self.folder.iterdir():
            if file.suffix == ".tiff" and "moved" in file.name:
                parts = file.stem.split("_")
                if len(parts) >= 4:
                    key = f"{parts[0]}_{parts[1]}_{parts[2]}"
                    if key not in seen:
                        self.path_list.append(file)
                        seen.add(key)

        self.total_paths = len(self.path_list)

    def build_interface(self):
        image_path = self.path_list[self.path_index.get()]
        width, height = Image.open(image_path).size
    
        self.curve_widget = CurveWidget(self.root, points=[], width=width, height=height, bg="#C8C8C8", path=image_path)
        self.curve_widget.grid(row=0, column=0)
    
        self.del_btn = tk.Button(self.root, width=20, text="DELETE POINTS", font="none 12", command=self.curve_widget.delete_el)
        self.del_btn.grid(row=1, columnspan=2)
    
        self.create_btn = tk.Button(self.root, width=20, text="CREATE MASK", font="none 12", command=lambda: self.create_mask(self.curve_widget))
        self.create_btn.grid(row=2, columnspan=2)
    
        self.save_btn = tk.Button(self.root, width=20, text="SAVE MASK", font="none 12", command=self.save_mask)
        self.save_btn.grid(row=3, columnspan=2)
    
        self.next_btn = tk.Button(self.root, width=20, text="NEXT IMAGE", font="none 12", command=self.next_image)
        self.next_btn.grid(row=4, columnspan=2)

    def on_format_change(self, value):
        self.selected_format = value
        
    def create_mask(self, curve_widget):
        points = chaikin_smooth(curve_widget.points)
    
        # Close the curve if the first and last points are far apart
        if points and (abs(points[0][0] - points[-1][0]) > 5 or abs(points[0][1] - points[-1][1]) > 5):
            points.append(points[0])
    
        mask = np.zeros((curve_widget.height, curve_widget.width), dtype=np.uint8)
    
        # Convert points to the required format
        contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    
        # Draw a closed polyline
        cv2.polylines(mask, [contour], isClosed=True, color=255, thickness=1)
    
        # Fill from an external point (0, 0)
        flood = mask.copy()
        h, w = flood.shape
        mask2 = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(flood, mask2, (0, 0), 128)
    
        # Inside the contour will be 0 (black), outside — 128, contour itself — 255
        # Extract inner region: where flood == 0
        result_mask = np.where(flood == 0, 1, 0).astype(np.uint8)
    
        self.mask = result_mask
    
        fig = Figure(figsize=(3, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax.imshow(self.mask)
    
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, self.root)
        self.canvas.get_tk_widget().grid(row=0, column=1)
    
    def save_mask(self):
        if self.save_path.get() == "no":
            selected = filedialog.askdirectory(title="Select folder to save masks")
            self.save_path.set(selected or "no")
        if self.save_path.get() == "no":
            return
    
        current_path = self.path_list[self.path_index.get()]
        parts = current_path.stem.split("_")
        mouse, exp, protocol, state = parts[0], parts[1], parts[2], parts[3]
    
        save_dir = Path(self.save_path.get()) / "Mask_draw"
        save_dir.mkdir(exist_ok=True)
    
        format_selected = self.selected_format
    
        base_name = f"{mouse}_{exp}_{protocol}_{state}_mask_draw"
    
        if format_selected == "hdf5":
            file_path = save_dir / f"{base_name}.hdf5"
            with h5py.File(file_path, "w") as f:
                f.create_dataset("Mask_draw", data=self.mask * 255)
            print(f"Mask saved to: {file_path}")
    
        elif format_selected in ["png", "tiff", "bmp"]:
            file_path = save_dir / f"{base_name}.{format_selected}"
            cv2.imwrite(str(file_path), self.mask * 255)
            print(f"Mask saved to: {file_path}")
    
        else:
            print("Unsupported format selected.")

    def next_image(self):
        self.path_index.set(self.path_index.get() + 1)
        if self.path_index.get() >= self.total_paths:
            self.curve_widget.grid_remove()
            if self.canvas:
                self.canvas.get_tk_widget().grid_remove()
            tk.Label(self.root, text="That is all!").grid(row=0, columnspan=2)
            return

        # Clear GUI
        self.curve_widget.grid_remove()
        if self.canvas:
            self.canvas.get_tk_widget().grid_remove()
        self.del_btn.grid_remove()
        self.create_btn.grid_remove()
        self.save_btn.grid_remove()
        self.next_btn.grid_remove()

        self.build_interface()
    
def run_gui():
    root = tk.Tk()
    app = MaskDrawerApp(root)
    root.mainloop()