import numpy as np
from paz.datasets import CityScapes
from pipelines import PreprocessSegmentationIds
from pipelines import PostprocessSegmentationIds
from pipelines import PostProcessImage


if __name__ == "__main__":
    from paz.backend.image import show_image

    label_path = '/home/octavio/Downloads/dummy/gtFine/'
    # label_path = '/home/octavio/Downloads/dummy/gtCoarse/'
    image_path = '/home/octavio/Downloads/dummy/RGB_images/leftImg8bit/'
    data_manager = CityScapes(image_path, label_path, 'train')
    dataset = data_manager.load_data()
    class_names = data_manager.class_names
    num_classes = len(class_names)
    preprocess = PreprocessSegmentationIds((128, 128), num_classes)
    postprocess_masks = PostprocessSegmentationIds(num_classes)
    postprocess_image = PostProcessImage()
    for sample in dataset:
        preprocessed_sample = preprocess(sample)
        image = preprocessed_sample['inputs']['input_1']
        image = postprocess_image(image)
        masks = preprocessed_sample['labels']['masks']
        masks = postprocess_masks(masks)
        mask_and_image = np.concatenate([masks, image], axis=1)
        show_image(mask_and_image)
