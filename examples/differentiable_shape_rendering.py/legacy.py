def Render(image_shape, world_to_camera, rays, shadows=False):
    ray_origins, ray_directions = rays
    H, W = image_shape

    def compute_shape_shadow_mask(
        scene, masks, lights, ray_origins, ray_directions, shape, mask
    ):
        hit_mask, depth, points, normals, eyes = intersect_shape(
            shape, ray_origins, ray_directions
        )

        hit_mask = jp.where(mask, hit_mask, False)
        depth = jp.where(jp.expand_dims(mask, axis=1), depth, 1e6)
        points = jp.where(jp.expand_dims(mask, axis=1), points, 1e6)
        points = move_toward_normals(points, normals)

        shadow_masks = []
        for light in lights:
            points_to_light = light.position - points
            points_to_light_distances = jp.linalg.norm(
                points_to_light, axis=1, keepdims=True
            )
            points_to_light = points_to_light / points_to_light_distances
            intersect = partial(
                intersect_shape,
                ray_origins=points,
                ray_directions=points_to_light,
            )
            _hit_masks, scene_depths, scene_points3D, _normals, _ = jax.vmap(
                intersect, in_axes=0
            )(scene)

            # I have to mask all _hit_masks and depths using the generic mask
            _hit_masks = jp.where(
                jp.expand_dims(masks, axis=1), _hit_masks, False
            )
            scene_depths = jp.where(
                jp.expand_dims(masks, axis=[1, 2]), scene_depths, 1e6
            )

            scene_depth = jp.min(scene_depths, axis=0)
            scene_hit_mask = compute_scene_hit_mask(_hit_masks)
            points_to_light_distances = jp.squeeze(
                points_to_light_distances, axis=1
            )
            scene_depth = jp.squeeze(scene_depth, axis=1)
            first_hit_light_source = points_to_light_distances > scene_depth
            is_shadow = jp.logical_and(scene_hit_mask, first_hit_light_source)
            shadow_masks.append(is_shadow.astype(jp.int32))
        shadow_masks = jp.array(shadow_masks)
        return hit_mask, depth, shadow_masks, points, normals

    def intersect(shape):
        return intersect_shape(shape, ray_origins, ray_directions)

    def render_with_shadows(scene, mask, lights):
        shape_shadow_mask = partial(
            compute_shape_shadow_mask,
            scene,
            mask,
            lights,
            ray_origins,
            ray_directions,
        )
        mask = jp.expand_dims(mask, axis=1)
        hit_masks, depths, shadow_masks, points, normals = jax.vmap(
            shape_shadow_mask
        )(scene, mask)
        eyes = compute_eyes(ray_directions)

        def _color(shape, points, normals, shadow_mask):
            color_by_lights = []
            for light_arg, light in enumerate(lights):
                shadow_light_mask = shadow_mask[light_arg, :]
                color = compute_colors_with_shadow(
                    shape,
                    shape.material,
                    points,
                    normals,
                    eyes,
                    light,
                    shadow_light_mask,
                )
                color_by_lights.append(color)
            return jp.array(color_by_lights)

        colors_per_light = jax.vmap(_color)(
            scene, points, normals, shadow_masks
        )
        colors = jp.sum(colors_per_light, axis=1)
        return postprocess(
            hit_masks,
            depths,
            colors,
            world_to_camera,
            ray_origins,
            ray_directions,
            H,
            W,
        )

    def render_without_shadows(scene, mask, lights):
        _render = partial(
            render_shape,
            lights=lights,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
        )
        hit_masks, depths, colors = jax.vmap(_render, in_axes=0)(scene)
        mask = jp.expand_dims(mask, 1)
        hit_masks = jp.where(mask, hit_masks, False)
        depths = jp.where(jp.expand_dims(mask, 1), depths, 1e6)
        return postprocess(
            hit_masks,
            depths,
            colors,
            world_to_camera,
            ray_origins,
            ray_directions,
            H,
            W,
        )

    return render_with_shadows if shadows else render_without_shadows
