from collections import namedtuple

import cv2
import numpy as np

import paz

POINT = "point"
BOX = "box"
POSITIVE = 1
NEGATIVE = 0
NUM_CLASS_COLORS = 10
CLASS_COLOR_STEP = 7


def build_class_colors():
    colors = paz.draw.lincolor(NUM_CLASS_COLORS, saturation=1.0)
    class_colors = []
    for class_arg in range(NUM_CLASS_COLORS):
        color_arg = (class_arg * CLASS_COLOR_STEP) % NUM_CLASS_COLORS
        class_colors.append(colors[color_arg])
    return tuple(class_colors)


CLASS_COLORS = build_class_colors()
Prompt = namedtuple("Prompt", "class_arg kind coordinates label")
ClassPrompt = namedtuple("ClassPrompt", "class_arg points labels box color")


class PromptSelector:
    def __init__(self, image, segment, window_name="SAM 2 prompts"):
        self.image = np.array(image, copy=True)
        self.segment = segment
        self.window_name = window_name
        self.result = np.array(image, copy=True)
        self.prompts = ()
        self.clicks = []
        self.mode = POINT
        self.label = POSITIVE
        self.active_class = 0
        self.corner = None

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.record_click)
        should_quit = False
        while not should_quit:
            while self.clicks:
                self.add_click(self.clicks.pop(0))
            interface = self.draw_interface()
            paz.image.show(interface, self.window_name, False)
            key = cv2.waitKey(20) & 0xFF
            should_quit = self.update(key)
            args = self.window_name, cv2.WND_PROP_VISIBLE
            visible = cv2.getWindowProperty(*args)
            should_quit = should_quit or visible < 1
        cv2.destroyWindow(self.window_name)
        return self.draw_prompts(self.result)

    def record_click(self, event, x, y, flags, data):
        height, width = self.image.shape[:2]
        within_image = x < width and y < height
        if event == cv2.EVENT_LBUTTONDOWN and within_image:
            self.clicks.append((x, y))

    def add_click(self, point):
        if self.mode == POINT:
            prompt = Prompt(self.active_class, POINT, point, self.label)
            self.prompts = self.prompts + (prompt,)
            self.apply()
        elif self.corner is None:
            self.corner = point
        else:
            self.add_box(point)
            self.apply()

    def update(self, key):
        should_quit = False
        if key == ord("p"):
            self.select_point(POSITIVE)
        elif key == ord("n"):
            self.select_point(NEGATIVE)
        elif key == ord("b"):
            self.mode = BOX
        elif key == ord("c"):
            self.start_class()
        elif key == ord("u"):
            self.undo()
        elif key == ord("s"):
            self.apply()
        elif key == ord("q"):
            should_quit = True
        return should_quit

    def apply(self):
        prompts = self.build_class_prompts()
        if prompts:
            self.result = self.segment(prompts)
        else:
            self.result = np.array(self.image, copy=True)

    def build_class_prompts(self):
        class_args = sorted({prompt.class_arg for prompt in self.prompts})
        class_prompts = []
        for class_arg in class_args:
            points, labels, box = self.unpack_class(class_arg)
            color = self.get_color(class_arg)
            prompt = ClassPrompt(class_arg, points, labels, box, color)
            class_prompts.append(prompt)
        return tuple(class_prompts)

    def draw_interface(self):
        prompt_image = self.draw_prompts(self.result)
        height, width, channels = prompt_image.shape
        footer = np.zeros((92, width, channels), dtype=prompt_image.dtype)
        image = np.concatenate([prompt_image, footer], axis=0)
        help_text = "p positive | n negative | b box | c new class"
        point = (8, height + 21)
        args = image, help_text, point, 0.48, paz.draw.WHITE
        image = self.draw_text(*args)
        help_text = "u undo | s apply | q quit"
        point = (8, height + 46)
        args = image, help_text, point, 0.48, paz.draw.WHITE
        image = self.draw_text(*args)
        status, color = self.build_status()
        return self.draw_text(image, status, (8, height + 75), 0.55, color)

    def draw_text(self, image, message, point, scale, color):
        args = image, message, point, cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(*args, scale, color, 1, cv2.LINE_AA)
        return image

    def draw_prompts(self, image):
        image = np.array(image, copy=True)
        for prompt in self.prompts:
            color = self.get_color(prompt.class_arg)
            if prompt.kind == BOX:
                image = paz.draw.box(image, prompt.coordinates, color, 3)
            elif prompt.label == POSITIVE:
                image = paz.draw.keypoint(image, prompt.coordinates, color, 7)
            else:
                image = self.draw_negative(image, prompt.coordinates, color)
        if self.corner is not None:
            color = self.get_color(self.active_class)
            image = paz.draw.keypoint(image, self.corner, color, 7)
        return image

    def draw_negative(self, image, point, color):
        x, y = point
        points = (
            ((x - 6, y - 6), (x + 6, y + 6)),
            ((x - 6, y + 6), (x + 6, y - 6)),
        )
        for point_0, point_1 in points:
            image = paz.draw.line(image, point_0, point_1, (0, 0, 0), 5)
            image = paz.draw.line(image, point_0, point_1, color, 3)
        return image

    def add_box(self, point):
        box = order_box(self.corner, point)
        prompt = Prompt(self.active_class, BOX, box, None)
        self.prompts = self.prompts + (prompt,)
        self.corner = None

    def select_point(self, label):
        self.mode = POINT
        self.label = label
        self.corner = None

    def start_class(self):
        self.active_class = self.next_class_arg()
        self.corner = None

    def undo(self):
        if self.corner is not None:
            self.corner = None
        elif self.prompts:
            self.prompts = self.prompts[:-1]
            self.active_class = self.last_class_arg()
            self.apply()

    def unpack_class(self, class_arg):
        points, labels, box = [], [], None
        for prompt in self.prompts:
            if prompt.class_arg != class_arg:
                continue
            if prompt.kind == POINT:
                points.append(prompt.coordinates)
                labels.append(prompt.label)
            else:
                box = prompt.coordinates
        points = points if points else None
        labels = labels if labels else None
        return points, labels, box

    def build_status(self):
        num_classes = len({prompt.class_arg for prompt in self.prompts})
        if self.mode == BOX:
            class_arg = self.active_class
            mode = "box: click two corners"
        elif self.label == POSITIVE:
            class_arg = self.active_class
            mode = "positive point"
        else:
            class_arg = self.active_class
            mode = "negative point"
        status = f"Class {class_arg + 1}: {mode} | classes: {num_classes}"
        return status, self.get_color(class_arg)

    def get_color(self, class_arg):
        return CLASS_COLORS[class_arg % NUM_CLASS_COLORS]

    def next_class_arg(self):
        class_args = [prompt.class_arg for prompt in self.prompts]
        class_arg = max(class_args) + 1 if class_args else 0
        return class_arg

    def last_class_arg(self):
        class_arg = self.prompts[-1].class_arg if self.prompts else 0
        return class_arg


def order_box(point_0, point_1):
    x_0, y_0 = point_0
    x_1, y_1 = point_1
    return min(x_0, x_1), min(y_0, y_1), max(x_0, x_1), max(y_0, y_1)
