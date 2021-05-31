# Generate images for PoseCNN

"""
        # Generate the semantic segmentation image
        image_masks, image_masks_3d = list(), list()

        for color in self.colors:
            image_mask = np.apply_along_axis(lambda x: int(np.array_equal(x, color)), 2, image_ambient_light)
            image_masks.append(image_mask)
            image_masks_3d.append(np.repeat(image_mask[:, :, np.newaxis], 3, axis=2))

        distances_x_direction, distances_y_direction, depth_images = list(), list(), list()
        # Generate the images with centers in x and y direction
        for image_mask, object_center in zip(image_masks, object_centers):
            image_pixel_values = np.flip(np.array(np.meshgrid(np.arange(0, self.viewport_size[0], 1), np.arange(0, self.viewport_size[1], 1))).T, axis=2)
            image_pixel_values = image_pixel_values*np.reshape(image_mask, (self.viewport_size[0], self.viewport_size[1], 1))
            distance_x_direction = np.apply_along_axis(lambda x: (x[0] - object_center[0])/np.linalg.norm(x - object_center[:2]), 2, image_pixel_values)*image_mask
            distance_y_direction = np.apply_along_axis(lambda x: (x[1] - object_center[1]) / np.linalg.norm(x - object_center[:2]), 2, image_pixel_values)*image_mask

            # Generate depth image
            depth_image = image_mask*object_center[2]

            distances_x_direction.append(distance_x_direction)
            distances_y_direction.append(distance_y_direction)
            depth_images.append(depth_image)

        distance_x_direction = reduce(lambda x, y: x + y, distances_x_direction)
        distance_y_direction = reduce(lambda x, y: x + y, distances_y_direction)
        depth_image = reduce(lambda x, y: x + y, depth_images)
        """