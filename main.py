import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

Image.MAX_IMAGE_PIXELS = None

globals()["scale_factor"] = 1

# background
# low_red_default = [165, 5, 5]
# upper_red_default = [180, 255, 255]
#
# low_blue_default = [100, 2, 50]
# upper_blue_default = [180, 255, 255]
#
# median_range = 151
# kernel_radius_default = 9

# blue part
low_red_default = [164, 5, 5]
upper_red_default = [180, 255, 255]

low_blue_default = [100, 2, 50]
upper_blue_default = [165, 255, 255]

median_range = 39
kernel_radius_default = 31

chunk_size = 10000
overlay_size = 1024


def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img_original = Image.open(file_path)
        if 'img_original' in globals():
            globals()['img_original'] = None
        if 'img_filtered' in globals():
            globals()['img_filtered'] = None
        globals()['img_original'] = img_original
        globals()['img_original_backup'] = img_original.copy()
        apply_changes()


def save_image():
    if 'img_filtered' in globals():
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if file_path:
            original_image = globals()['img_filtered'].copy()
            original_image.save(file_path)


def apply_filters(image, lower_red=None, lower_blue=None, kernel_radius_=kernel_radius_default):
    if lower_red is None:
        lower_red = low_red_default
    if lower_blue is None:
        lower_blue = low_blue_default

    img_array = np.array(image)
    img_median = cv2.medianBlur(img_array, median_range)
    hsv_image = cv2.cvtColor(img_median, cv2.COLOR_RGB2HSV)

    upper_red = np.array(upper_red_default)
    mask_red = cv2.inRange(hsv_image, np.array(lower_red), upper_red)

    upper_blue = np.array(upper_blue_default)
    mask_blue = cv2.inRange(hsv_image, np.array(lower_blue), upper_blue)

    mask_not_red = cv2.bitwise_not(mask_red)
    mask_blue_not_red = cv2.bitwise_and(mask_blue, mask_not_red)

    mask_final = mask_blue_not_red

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_radius_, kernel_radius_))
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel, iterations=2)

    segment = cv2.bitwise_and(img_array, img_array, mask=mask_final)
    img_segmented = Image.fromarray(segment)
    globals()['img_filtered'] = img_segmented


def apply_changes():
    lower_red = [int(entry_lower_red_hue.get()), int(entry_lower_red_saturation.get()),
                 int(entry_lower_red_value.get())]
    lower_blue = [int(entry_lower_blue_hue.get()), int(entry_lower_blue_saturation.get()),
                  int(entry_lower_blue_value.get())]
    median_range_ = int(entry_median_range.get())
    if median_range_ % 2 == 1:
        median_range_ += 1
    kernel_radius_ = int(entry_kernel_radius.get())
    globals()["scale_factor"] = scale.get() / 100
    if 'img_original' in globals():
        apply_filters(image=globals()['img_original_backup'].copy(), lower_red=lower_red, lower_blue=lower_blue,
                      kernel_radius_=kernel_radius_)
        update_previews()


def update_previews():
    if 'img_original' in globals():
        img_original_thumbnail = globals()['img_original'].copy()
        img_original_thumbnail.thumbnail((600 * globals()["scale_factor"], 600 * globals()["scale_factor"]),
                                         Image.Resampling.LANCZOS)
        update_preview(img_label_original, img_original_thumbnail)

    if 'img_filtered' in globals():
        img_filtered_thumbnail = globals()['img_filtered'].copy()
        img_filtered_thumbnail.thumbnail((600 * globals()["scale_factor"], 600 * globals()["scale_factor"]),
                                         Image.Resampling.LANCZOS)
        update_preview(img_label_processed, img_filtered_thumbnail)


def update_preview(img_label, resized_image):
    img_tk = ImageTk.PhotoImage(resized_image)
    img_label.config(image=img_tk)
    img_label.image = img_tk


def mapping():
    filter_image = globals()['img_filtered'].copy()
    filter_image_ = np.array(filter_image).copy()
    gray_image = cv2.cvtColor(filter_image_, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contoured_image = globals()['img_original']
    contoured_image_ = np.array(contoured_image).copy()
    cv2.drawContours(contoured_image_, contours, -1, (0, 255, 0), 2)
    globals()["img_original"] = Image.fromarray(contoured_image_)
    apply_changes()


def split_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        _overlap_size = overlay_size  # Overlap between chunks
        _chunk_size = chunk_size  # Size of each chunk
        output_folder = filedialog.askdirectory()

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        width, height = image.size
        count = 0

        for i in range(0, width, _chunk_size - _overlap_size):
            for j in range(0, height, _chunk_size - _overlap_size):
                box = (i, j, i + _chunk_size, j + _chunk_size)
                chunk = image.crop(box)
                chunk.save(os.path.join(output_folder, f'chunk_{count}.png'))
                count += 1


# Create main window
root = tk.Tk()
root.title("Image Processing")
root.geometry('1920x1080')

img_frame = tk.Frame(root)
img_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

# Display Original Image
img_label_original = tk.Label(img_frame)
img_label_original.image = None
img_label_original.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

# Display Processed Image
img_label_processed = tk.Label(img_frame)
img_label_processed.image = None
img_label_processed.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

# Buttons
split_image_btn = tk.Button(root, text="Split Image", command=split_image)
split_image_btn.pack(side=tk.TOP, anchor=tk.N, pady=10)

upload_btn = tk.Button(root, text="Upload Image", command=upload_image)
upload_btn.pack(side=tk.TOP, anchor=tk.N, pady=10)

save_btn = tk.Button(root, text="Save Image", command=save_image)
save_btn.pack(side=tk.TOP, anchor=tk.N, pady=10)

# Scale
scale = tk.Scale(root, from_=10, to=125, orient="horizontal", label="Scale Factor (%)")
scale.set(100)
scale.pack(side=tk.TOP, anchor=tk.N, pady=10)

# Entry for Red Mask Bounds
entry_lower_red_hue = tk.Entry(root)
entry_lower_red_hue.insert(0, str(low_red_default[0]))
label_lower_red_hue = tk.Label(root, text="Lower Red Hue:")
label_lower_red_hue.pack(side=tk.TOP, anchor=tk.N)
entry_lower_red_hue.pack(side=tk.TOP, anchor=tk.N)

entry_lower_red_saturation = tk.Entry(root)
entry_lower_red_saturation.insert(0, str(low_red_default[1]))
label_lower_red_saturation = tk.Label(root, text="Lower Red Saturation:")
label_lower_red_saturation.pack(side=tk.TOP, anchor=tk.N)
entry_lower_red_saturation.pack(side=tk.TOP, anchor=tk.N)

entry_lower_red_value = tk.Entry(root)
entry_lower_red_value.insert(0, str(low_red_default[2]))
label_lower_red_value = tk.Label(root, text="Lower Red Value:")
label_lower_red_value.pack(side=tk.TOP, anchor=tk.N)
entry_lower_red_value.pack(side=tk.TOP, anchor=tk.N)

# Entry for Blue Mask Bounds
entry_lower_blue_hue = tk.Entry(root)
entry_lower_blue_hue.insert(0, str(low_blue_default[0]))
label_lower_blue_hue = tk.Label(root, text="Lower Blue Hue:")
label_lower_blue_hue.pack(side=tk.TOP, anchor=tk.N)
entry_lower_blue_hue.pack(side=tk.TOP, anchor=tk.N)

entry_lower_blue_saturation = tk.Entry(root)
entry_lower_blue_saturation.insert(0, str(low_blue_default[1]))
label_lower_blue_saturation = tk.Label(root, text="Lower Blue Saturation:")
label_lower_blue_saturation.pack(side=tk.TOP, anchor=tk.N)
entry_lower_blue_saturation.pack(side=tk.TOP, anchor=tk.N)

entry_lower_blue_value = tk.Entry(root)
entry_lower_blue_value.insert(0, str(low_blue_default[2]))
label_lower_blue_value = tk.Label(root, text="Lower Blue Value:")
label_lower_blue_value.pack(side=tk.TOP, anchor=tk.N)
entry_lower_blue_value.pack(side=tk.TOP, anchor=tk.N)

# Entry for Median Range
entry_median_range = tk.Entry(root)
entry_median_range.insert(0, str(median_range))
label_median_range = tk.Label(root, text="Median Range(odd):")
label_median_range.pack(side=tk.TOP, anchor=tk.N)
entry_median_range.pack(side=tk.TOP, anchor=tk.N)

# Entry for Kernel Radius
entry_kernel_radius = tk.Entry(root)
entry_kernel_radius.insert(0, str(kernel_radius_default))
label_kernel_radius = tk.Label(root, text="Kernel Radius:")
label_kernel_radius.pack(side=tk.TOP, anchor=tk.N)
entry_kernel_radius.pack(side=tk.TOP, anchor=tk.N)

# Apply Change Button
apply_change_btn = tk.Button(root, text="Apply Change", command=apply_changes)
apply_change_btn.pack(side=tk.TOP, anchor=tk.N, pady=10)

# Mapping Button
mapping_btn = tk.Button(root, text="Mapping", command=mapping)
mapping_btn.pack(side=tk.TOP, anchor=tk.N, pady=10)

root.mainloop()

