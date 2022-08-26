import numpy as np
from backend import draw_box
from paz.datasets import Shapes

if __name__ == '__main__':
    from paz.backend.image import show_image
    data_manager = Shapes(1000, (128, 128), iou_thresh=0.3, max_num_shapes=3)
    dataset = data_manager.load_data()
    for sample in dataset:
        image = sample['image']
        masks = (sample['masks'] * 255.0).astype('uint8')
        background_mask, masks = masks[..., 0:1], masks[..., 1:]
        background_mask = np.repeat(background_mask, 3, axis=-1)
        boxes = sample['box_data']
        for box in boxes:
            coordinates, class_arg = box[:4], box[4]
            # coordinates = denormalize_box(coordinates, (128, 128))
            class_name = data_manager.arg_to_name[class_arg]
            image = draw_box(image, coordinates, class_name, 1.0)
        show_image(np.concatenate([image, masks, background_mask], axis=1))
