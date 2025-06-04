import jax.numpy as jp
import paz


def EstimatePoseKeypoints(

    def call(self, image):
        boxes2D = self.detect(image)["boxes2D"]
        boxes2D = self.square(boxes2D)
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)
        poses6D, keypoints2D = [], []
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            keypoints = self.estimate_keypoints(cropped_image)["keypoints"]
            keypoints = self.change_coordinates(keypoints, box2D)
            pose6D = self.solve_PNP(keypoints)
            image = self.draw_keypoints(image, keypoints)
            image = self.draw_box(image, pose6D)
            keypoints2D.append(keypoints)
            poses6D.append(pose6D)
        return boxes2D, keypoints2D, poses6D

    def call(image):
        boxes = paz.detection.get_boxes(detect(image))
        boxes = paz.boxes.square(boxes)
        boxes = paz.boxes.scale(boxes, box_scale, box_scale)
        boxes = paz.cast(boxes, "int32")
        boxes = paz.boxes.remove_invalid(boxes)
        total_keypoints = []
        for box in boxes:
            keypoints = estimate_keypoints(paz.image.crop(image, box))
            keypoints = paz.points2D.shift_to_box_origin(keypoints, box)
            total_keypoints.append(keypoints)
        total_keypoints = jp.array(total_keypoints)
        return boxes, total_keypoints

    return (lambda x: (y := call(x), draw(x, *y))) if callable(draw) else call
