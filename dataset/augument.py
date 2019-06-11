"""
The data augmentation operations of the original SSD implementation.
Copyright (C) 2018 Pierluigi Ferrari
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cv2
import inspect
import numpy as np


class RandomScaleAugmentation:
    """
    Reproduces the data augmentation pipeline used in the training of the original
    Caffe implementation of SSD.
    """

    def __init__(self,
                 img_height=768,
                 img_width=1024,
                 labels_format={
                     'class_id': 0,
                     'xmin': 1,
                     'ymin': 2,
                     'xmax': 3,
                     'ymax': 4
                 }):
        """
        Arguments:
            height (int): The desired height of the output images in pixels.
            width (int): The desired width of the output images in pixels.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the
                background pixels of the translated images.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        """

        self.labels_format = labels_format
        self.random_crop = SSDRandomCrop(labels_format=self.labels_format)
        # This box filter makes sure that the resized images don't contain any degenerate boxes.
        # Resizing the images could lead the boxes to becomes smaller. For boxes that are already
        # pretty small, that might result in boxes with height and/or width zero, which we obviously
        # cannot allow.
        self.box_filter = BoxFilter(check_overlap=False,
                                    check_min_area=False,
                                    check_degenerate=True,
                                    labels_format=self.labels_format)

        self.resize = ResizeRandomInterp(height=img_height,
                                         width=img_width,
                                         interpolation_modes=[
                                             cv2.INTER_NEAREST,
                                             cv2.INTER_LINEAR, cv2.INTER_CUBIC,
                                             cv2.INTER_AREA, cv2.INTER_LANCZOS4
                                         ],
                                         box_filter=self.box_filter,
                                         labels_format=self.labels_format)

        self.sequence = [self.random_crop, self.resize]

    def __call__(self, image, labels, return_inverter=False):
        self.random_crop.labels_format = self.labels_format
        self.resize.labels_format = self.labels_format
        inverters = []
        for transform in self.sequence:
            if return_inverter and ('return_inverter' in inspect.signature(
                    transform).parameters):
                image, labels, inverter = transform(image,
                                                    labels,
                                                    return_inverter=True)
                inverters.append(inverter)
            else:
                image, labels = transform(image, labels)

        if return_inverter:
            return image, labels, inverters[::-1]
        else:
            return image, labels


class SSDRandomCrop:
    """
    Performs the same random crops as defined by the `batch_sampler` instructions
    of the original Caffe implementation of SSD. A description of this random cropping
    strategy can also be found in the data augmentation section of the paper:
    https://arxiv.org/abs/1512.02325
    """

    def __init__(self,
                 labels_format={
                     'class_id': 0,
                     'xmin': 1,
                     'ymin': 2,
                     'xmax': 3,
                     'ymax': 4
                 }):
        """
        Arguments:
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        """

        self.labels_format = labels_format

        # This randomly samples one of the lower IoU bounds defined
        # by the `sample_space` every time it is called.
        self.bound_generator = BoundGenerator(
            sample_space=((None, None), (0.1, None), (0.3, None), (0.5, None),
                          (0.7, None), (0.9, None)),
            weights=None)

        # Produces coordinates for candidate patches such that the height
        # and width of the patches are between 0.3 and 1.0 of the height
        # and width of the respective image and the aspect ratio of the
        # patches is between 0.5 and 2.0.
        self.patch_coord_generator = PatchCoordinateGenerator(
            must_match='h_w',
            min_scale=0.6,
            max_scale=1.0,
            scale_uniformly=False,
            min_aspect_ratio=0.5,
            max_aspect_ratio=2.0)

        # Filters out boxes whose center point does not lie within the
        # chosen patches.
        self.box_filter = BoxFilter(check_overlap=True,
                                    check_min_area=False,
                                    check_degenerate=False,
                                    overlap_criterion='center_point',
                                    labels_format=self.labels_format)

        # Determines whether a given patch is considered a valid patch.
        # Defines a patch to be valid if at least one ground truth bounding box
        # (n_boxes_min == 1) has an IoU overlap with the patch that
        # meets the requirements defined by `bound_generator`.
        self.image_validator = ImageValidator(overlap_criterion='iou',
                                              n_boxes_min=1,
                                              labels_format=self.labels_format,
                                              border_pixels='half')

        # Performs crops according to the parameters set in the objects above.
        # Runs until either a valid patch is found or the original input image
        # is returned unaltered. Runs a maximum of 50 trials to find a valid
        # patch for each new sampled IoU threshold. Every 50 trials, the original
        # image is returned as is with probability (1 - prob) = 0.143.
        self.random_crop = RandomPatchInf(
            patch_coord_generator=self.patch_coord_generator,
            box_filter=self.box_filter,
            image_validator=self.image_validator,
            bound_generator=self.bound_generator,
            n_trials_max=50,
            clip_boxes=True,
            prob=0.857,
            labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):
        self.random_crop.labels_format = self.labels_format
        return self.random_crop(image, labels, return_inverter)


class BoundGenerator:
    """
    Generates pairs of floating point values that represent lower and upper bounds
    from a given sample space.
    """

    def __init__(self,
                 sample_space=((0.1, None), (0.3, None), (0.5, None),
                               (0.7, None), (0.9, None), (None, None)),
                 weights=None):
        """
        Arguments:
            sample_space (list or tuple): A list, tuple, or array-like object of shape
                `(n, 2)` that contains `n` samples to choose from, where each sample
                is a 2-tuple of scalars and/or `None` values.
            weights (list or tuple, optional): A list or tuple representing the distribution
                over the sample space. If `None`, a uniform distribution will be assumed.
        """

        if (not (weights is None)) and len(weights) != len(sample_space):
            raise ValueError(
                "`weights` must either be `None` for uniform distribution or have the same length as `sample_space`."
            )

        self.sample_space = []
        for bound_pair in sample_space:
            if len(bound_pair) != 2:
                raise ValueError(
                    "All elements of the sample space must be 2-tuples.")
            bound_pair = list(bound_pair)
            if bound_pair[0] is None: bound_pair[0] = 0.0
            if bound_pair[1] is None: bound_pair[1] = 1.0
            if bound_pair[0] > bound_pair[1]:
                raise ValueError(
                    "For all sample space elements, the lower bound "
                    "cannot be greater than the upper bound.")
            self.sample_space.append(bound_pair)

        self.sample_space_size = len(self.sample_space)

        if weights is None:
            self.weights = [1.0 / self.sample_space_size
                            ] * self.sample_space_size
        else:
            self.weights = weights

    def __call__(self):
        """
        Returns:
            An item of the sample space, i.e. a 2-tuple of scalars.
        """
        i = np.random.choice(self.sample_space_size, p=self.weights)
        return self.sample_space[i]


class BoxFilter:
    """
    Returns all bounding boxes that are valid with respect to a the defined criteria.
    """

    def __init__(self,
                 check_overlap=True,
                 check_min_area=True,
                 check_degenerate=True,
                 overlap_criterion='center_point',
                 overlap_bounds=(0.3, 1.0),
                 min_area=10,
                 labels_format={
                     'class_id': 0,
                     'xmin': 1,
                     'ymin': 2,
                     'xmax': 3,
                     'ymax': 4
                 },
                 border_pixels='half'):
        """
        Arguments:
            check_overlap (bool, optional): Whether or not to enforce the overlap requirements defined by
                `overlap_criterion` and `overlap_bounds`. Sometimes you might want to use the box filter only
                to enforce a certain minimum area for all boxes (see next argument), in such cases you can
                turn the overlap requirements off.
            check_min_area (bool, optional): Whether or not to enforce the minimum area requirement defined
                by `min_area`. If `True`, any boxes that have an area (in pixels) that is smaller than `min_area`
                will be removed from the labels of an image. Bounding boxes below a certain area aren't useful
                training examples. An object that takes up only, say, 5 pixels in an image is probably not
                recognizable anymore, neither for a human, nor for an object detection model. It makes sense
                to remove such boxes.
            check_degenerate (bool, optional): Whether or not to check for and remove degenerate bounding boxes.
                Degenerate bounding boxes are boxes that have `xmax <= xmin` and/or `ymax <= ymin`. In particular,
                boxes with a width and/or height of zero are degenerate. It is obviously important to filter out
                such boxes, so you should only set this option to `False` if you are certain that degenerate
                boxes are not possible in your data and processing chain.
            overlap_criterion (str, optional): Can be either of 'center_point', 'iou', or 'area'. Determines
                which boxes are considered valid with respect to a given image. If set to 'center_point',
                a given bounding box is considered valid if its center point lies within the image.
                If set to 'area', a given bounding box is considered valid if the quotient of its intersection
                area with the image and its own area is within the given `overlap_bounds`. If set to 'iou', a given
                bounding box is considered valid if its IoU with the image is within the given `overlap_bounds`.
            overlap_bounds (list or BoundGenerator, optional): Only relevant if `overlap_criterion` is 'area' or 'iou'.
                Determines the lower and upper bounds for `overlap_criterion`. Can be either a 2-tuple of scalars
                representing a lower bound and an upper bound, or a `BoundGenerator` object, which provides
                the possibility to generate bounds randomly.
            min_area (int, optional): Only relevant if `check_min_area` is `True`. Defines the minimum area in
                pixels that a bounding box must have in order to be valid. Boxes with an area smaller than this
                will be removed.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
            border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
                Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
                to the boxes. If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong
                to the boxex, but not the other.
        """
        if not isinstance(overlap_bounds, (list, tuple, BoundGenerator)):
            raise ValueError(
                "`overlap_bounds` must be either a 2-tuple of scalars or a `BoundGenerator` object."
            )
        if isinstance(
                overlap_bounds,
            (list, tuple)) and (overlap_bounds[0] > overlap_bounds[1]):
            raise ValueError(
                "The lower bound must not be greater than the upper bound.")
        if not (overlap_criterion in {'iou', 'area', 'center_point'}):
            raise ValueError(
                "`overlap_criterion` must be one of 'iou', 'area', or 'center_point'."
            )
        self.overlap_criterion = overlap_criterion
        self.overlap_bounds = overlap_bounds
        self.min_area = min_area
        self.check_overlap = check_overlap
        self.check_min_area = check_min_area
        self.check_degenerate = check_degenerate
        self.labels_format = labels_format
        self.border_pixels = border_pixels

    def __call__(self, labels, image_height=None, image_width=None):
        """
        Arguments:
            labels (array): The labels to be filtered. This is an array with shape `(m,n)`, where
                `m` is the number of bounding boxes and `n` is the number of elements that defines
                each bounding box (box coordinates, class ID, etc.). The box coordinates are expected
                to be in the image's coordinate system.
            image_height (int): Only relevant if `check_overlap == True`. The height of the image
                (in pixels) to compare the box coordinates to.
            image_width (int): `check_overlap == True`. The width of the image (in pixels) to compare
                the box coordinates to.

        Returns:
            An array containing the labels of all boxes that are valid.
        """

        labels = np.copy(labels)

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        # Record the boxes that pass all checks here.
        requirements_met = np.ones(shape=labels.shape[0], dtype=np.bool)

        if self.check_degenerate:

            non_degenerate = (labels[:, xmax] > labels[:, xmin]) * (
                labels[:, ymax] > labels[:, ymin])
            requirements_met *= non_degenerate

        if self.check_min_area:

            min_area_met = (labels[:, xmax] - labels[:, xmin]) * (
                labels[:, ymax] - labels[:, ymin]) >= self.min_area
            requirements_met *= min_area_met

        if self.check_overlap:

            # Get the lower and upper bounds.
            if isinstance(self.overlap_bounds, BoundGenerator):
                lower, upper = self.overlap_bounds()
            else:
                lower, upper = self.overlap_bounds

            # Compute which boxes are valid.

            if self.overlap_criterion == 'iou':
                # Compute the patch coordinates.
                image_coords = np.array([0, 0, image_width, image_height])
                # Compute the IoU between the patch and all of the ground truth boxes.
                image_boxes_iou = iou(image_coords,
                                      labels[:, [xmin, ymin, xmax, ymax]],
                                      coords='corners',
                                      mode='element-wise',
                                      border_pixels=self.border_pixels)
                requirements_met *= (image_boxes_iou >
                                     lower) * (image_boxes_iou <= upper)

            elif self.overlap_criterion == 'area':
                if self.border_pixels == 'half':
                    d = 0
                elif self.border_pixels == 'include':
                    d = 1  # If border pixels are supposed to belong to the bounding boxes,
                    # we have to add one pixel to any difference `xmax - xmin` or `ymax - ymin`.
                elif self.border_pixels == 'exclude':
                    d = -1  # If border pixels are not supposed to belong to the bounding boxes,
                    # we have to subtract one pixel from any difference `xmax - xmin` or `ymax - ymin`.
                # Compute the areas of the boxes.
                box_areas = (labels[:, xmax] - labels[:, xmin] +
                             d) * (labels[:, ymax] - labels[:, ymin] + d)
                # Compute the intersection area between the patch and all of the ground truth boxes.
                clipped_boxes = np.copy(labels)
                clipped_boxes[:, [ymin, ymax]] = np.clip(
                    labels[:, [ymin, ymax]], a_min=0, a_max=image_height - 1)
                clipped_boxes[:, [xmin, xmax]] = np.clip(
                    labels[:, [xmin, xmax]], a_min=0, a_max=image_width - 1)
                intersection_areas = (
                    clipped_boxes[:, xmax] - clipped_boxes[:, xmin] + d) * (
                        clipped_boxes[:, ymax] - clipped_boxes[:, ymin] + d
                    )  # +1 because the border pixels belong to the box areas.
                # Check which boxes meet the overlap requirements.
                if lower == 0.0:
                    mask_lower = intersection_areas > lower * box_areas  # If `self.lower == 0`, we want to
                    # make sure that boxes with area 0 don't count, hence the ">" sign instead of the ">=" sign.
                else:
                    mask_lower = intersection_areas >= lower * box_areas  # Especially for the case `self.lower == 1`
                    # we want the ">=" sign, otherwise no boxes would count at all.
                mask_upper = intersection_areas <= upper * box_areas
                requirements_met *= mask_lower * mask_upper

            elif self.overlap_criterion == 'center_point':
                # Compute the center points of the boxes.
                cy = (labels[:, ymin] + labels[:, ymax]) / 2
                cx = (labels[:, xmin] + labels[:, xmax]) / 2
                # Check which of the boxes have center points within the cropped patch remove those that don't.
                requirements_met *= (cy >= 0.0) * (cy <= image_height - 1) * (
                    cx >= 0.0) * (cx <= image_width - 1)

        return labels[requirements_met]


class ImageValidator:
    """
    Returns `True` if a given minimum number of bounding boxes meets given overlap
    requirements with an image of a given height and width.
    """

    def __init__(self,
                 overlap_criterion='center_point',
                 bounds=(0.3, 1.0),
                 n_boxes_min=1,
                 labels_format={
                     'class_id': 0,
                     'xmin': 1,
                     'ymin': 2,
                     'xmax': 3,
                     'ymax': 4
                 },
                 border_pixels='half'):
        """
        Arguments:
            overlap_criterion (str, optional): Can be either of 'center_point', 'iou', or 'area'. Determines
                which boxes are considered valid with respect to a given image. If set to 'center_point',
                a given bounding box is considered valid if its center point lies within the image.
                If set to 'area', a given bounding box is considered valid if the quotient of its intersection
                area with the image and its own area is within `lower` and `upper`. If set to 'iou', a given
                bounding box is considered valid if its IoU with the image is within `lower` and `upper`.
            bounds (list or BoundGenerator, optional): Only relevant if `overlap_criterion` is 'area' or 'iou'.
                Determines the lower and upper bounds for `overlap_criterion`. Can be either a 2-tuple of scalars
                representing a lower bound and an upper bound, or a `BoundGenerator` object, which provides
                the possibility to generate bounds randomly.
            n_boxes_min (int or str, optional): Either a non-negative integer or the string 'all'.
                Determines the minimum number of boxes that must meet the `overlap_criterion` with respect to
                an image of the given height and width in order for the image to be a valid image.
                If set to 'all', an image is considered valid if all given boxes meet the `overlap_criterion`.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
            border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
                Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
                to the boxes. If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong
                to the boxex, but not the other.
        """
        if not ((isinstance(n_boxes_min, int) and n_boxes_min > 0)
                or n_boxes_min == 'all'):
            raise ValueError(
                "`n_boxes_min` must be a positive integer or 'all'.")
        self.overlap_criterion = overlap_criterion
        self.bounds = bounds
        self.n_boxes_min = n_boxes_min
        self.labels_format = labels_format
        self.border_pixels = border_pixels
        self.box_filter = BoxFilter(check_overlap=True,
                                    check_min_area=False,
                                    check_degenerate=False,
                                    overlap_criterion=self.overlap_criterion,
                                    overlap_bounds=self.bounds,
                                    labels_format=self.labels_format,
                                    border_pixels=self.border_pixels)

    def __call__(self, labels, image_height, image_width):
        """
        Arguments:
            labels (array): The labels to be tested. The box coordinates are expected
                to be in the image's coordinate system.
            image_height (int): The height of the image to compare the box coordinates to.
            image_width (int): The width of the image to compare the box coordinates to.

        Returns:
            A boolean indicating whether an imgae of the given height and width is
            valid with respect to the given bounding boxes.
        """

        self.box_filter.overlap_bounds = self.bounds
        self.box_filter.labels_format = self.labels_format

        # Get all boxes that meet the overlap requirements.
        valid_labels = self.box_filter(labels=labels,
                                       image_height=image_height,
                                       image_width=image_width)

        # Check whether enough boxes meet the requirements.
        if isinstance(self.n_boxes_min, int):
            # The image is valid if at least `self.n_boxes_min` ground truth boxes meet the requirements.
            if len(valid_labels) >= self.n_boxes_min:
                return True
            else:
                return False
        elif self.n_boxes_min == 'all':
            # The image is valid if all ground truth boxes meet the requirements.
            if len(valid_labels) == len(labels):
                return True
            else:
                return False


class RandomPatchInf:
    """
    Randomly samples a patch from an image. The randomness refers to whatever
    randomness may be introduced by the patch coordinate generator, the box filter,
    and the patch validator.

    Input images may be cropped and/or padded along either or both of the two
    spatial dimensions as necessary in order to obtain the required patch.

    This operation is very similar to `RandomPatch`, except that:
    1. This operation runs indefinitely until either a valid patch is found or
       the input image is returned unaltered, i.e. it cannot fail.
    2. If a bound generator is given, a new pair of bounds will be generated
       every `n_trials_max` iterations.
    """

    def __init__(self,
                 patch_coord_generator,
                 box_filter=None,
                 image_validator=None,
                 bound_generator=None,
                 n_trials_max=50,
                 clip_boxes=True,
                 prob=0.857,
                 background=(0, 0, 0),
                 labels_format={
                     'class_id': 0,
                     'xmin': 1,
                     'ymin': 2,
                     'xmax': 3,
                     'ymax': 4
                 }):
        """
        Arguments:
            patch_coord_generator (PatchCoordinateGenerator): A `PatchCoordinateGenerator` object
                to generate the positions and sizes of the patches to be sampled from the input images.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            image_validator (ImageValidator, optional): Only relevant if ground truth bounding boxes are given.
                An `ImageValidator` object to determine whether a sampled patch is valid. If `None`,
                any outcome is valid.
            bound_generator (BoundGenerator, optional): A `BoundGenerator` object to generate upper and
                lower bound values for the patch validator. Every `n_trials_max` trials, a new pair of
                upper and lower bounds will be generated until a valid patch is found or the original image
                is returned. This bound generator overrides the bound generator of the patch validator.
            n_trials_max (int, optional): Only relevant if ground truth bounding boxes are given.
                The sampler will run indefinitely until either a valid patch is found or the original image
                is returned, but this determines the maxmial number of trials to sample a valid patch for each
                selected pair of lower and upper bounds before a new pair is picked.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the
                sampled patch.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the potential
                background pixels of the scaled images. In the case of single-channel images,
                the first element of `background` will be used as the background pixel value.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        """

        if not isinstance(patch_coord_generator, PatchCoordinateGenerator):
            raise ValueError(
                "`patch_coord_generator` must be an instance of `PatchCoordinateGenerator`."
            )
        if not (isinstance(image_validator, ImageValidator)
                or image_validator is None):
            raise ValueError(
                "`image_validator` must be either `None` or an `ImageValidator` object."
            )
        if not (isinstance(bound_generator, BoundGenerator)
                or bound_generator is None):
            raise ValueError(
                "`bound_generator` must be either `None` or a `BoundGenerator` object."
            )
        self.patch_coord_generator = patch_coord_generator
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.bound_generator = bound_generator
        self.n_trials_max = n_trials_max
        self.clip_boxes = clip_boxes
        self.prob = prob
        self.background = background
        self.labels_format = labels_format
        self.sample_patch = CropPad(patch_ymin=None,
                                    patch_xmin=None,
                                    patch_height=None,
                                    patch_width=None,
                                    clip_boxes=self.clip_boxes,
                                    box_filter=self.box_filter,
                                    background=self.background,
                                    labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]
        self.patch_coord_generator.img_height = img_height
        self.patch_coord_generator.img_width = img_width

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        # Override the preset labels format.
        if not self.image_validator is None:
            self.image_validator.labels_format = self.labels_format
        self.sample_patch.labels_format = self.labels_format

        while True:  # Keep going until we either find a valid patch or return the original image.

            p = np.random.uniform(0, 1)
            if p >= (1.0 - self.prob):

                # In case we have a bound generator, pick a lower and upper bound for the patch validator.
                if not ((self.image_validator is None) or
                        (self.bound_generator is None)):
                    self.image_validator.bounds = self.bound_generator()

                # Use at most `self.n_trials_max` attempts to find a crop
                # that meets our requirements.
                for _ in range(max(1, self.n_trials_max)):

                    # Generate patch coordinates.
                    patch_ymin, patch_xmin, patch_height, patch_width = self.patch_coord_generator(
                    )

                    self.sample_patch.patch_ymin = patch_ymin
                    self.sample_patch.patch_xmin = patch_xmin
                    self.sample_patch.patch_height = patch_height
                    self.sample_patch.patch_width = patch_width

                    # Check if the resulting patch meets the aspect ratio requirements.
                    aspect_ratio = patch_width / patch_height
                    if not (self.patch_coord_generator.min_aspect_ratio <=
                            aspect_ratio <=
                            self.patch_coord_generator.max_aspect_ratio):
                        continue

                    if (labels is None) or (self.image_validator is None):
                        # We either don't have any boxes or if we do, we will accept any outcome as valid.
                        return self.sample_patch(image, labels,
                                                 return_inverter)
                    else:
                        # Translate the box coordinates to the patch's coordinate system.
                        new_labels = np.copy(labels)
                        new_labels[:, [ymin, ymax]] -= patch_ymin
                        new_labels[:, [xmin, xmax]] -= patch_xmin
                        # Check if the patch contains the minimum number of boxes we require.
                        if self.image_validator(labels=new_labels,
                                                image_height=patch_height,
                                                image_width=patch_width):
                            return self.sample_patch(image, labels,
                                                     return_inverter)
            else:
                if return_inverter:

                    def inverter(labels):
                        return labels

                if labels is None:
                    if return_inverter:
                        return image, inverter
                    else:
                        return image
                else:
                    if return_inverter:
                        return image, labels, inverter
                    else:
                        return image, labels


class PatchCoordinateGenerator:
    """
    Generates random patch coordinates that meet specified requirements.
    """

    def __init__(self,
                 img_height=None,
                 img_width=None,
                 must_match='h_w',
                 min_scale=0.3,
                 max_scale=1.0,
                 scale_uniformly=False,
                 min_aspect_ratio=0.5,
                 max_aspect_ratio=2.0,
                 patch_ymin=None,
                 patch_xmin=None,
                 patch_height=None,
                 patch_width=None,
                 patch_aspect_ratio=None):
        """
        Arguments:
            img_height (int): The height of the image for which the patch coordinates
                shall be generated. Doesn't have to be known upon construction.
            img_width (int): The width of the image for which the patch coordinates
                shall be generated. Doesn't have to be known upon construction.
            must_match (str, optional): Can be either of 'h_w', 'h_ar', and 'w_ar'.
                Specifies which two of the three quantities height, width, and aspect
                ratio determine the shape of the generated patch. The respective third
                quantity will be computed from the other two. For example,
                if `must_match == 'h_w'`, then the patch's height and width will be
                set to lie within [min_scale, max_scale] of the image size or to
                `patch_height` and/or `patch_width`, if given. The patch's aspect ratio
                is the dependent variable in this case, it will be computed from the
                height and width. Any given values for `patch_aspect_ratio`,
                `min_aspect_ratio`, or `max_aspect_ratio` will be ignored.
            min_scale (float, optional): The minimum size of a dimension of the patch
                as a fraction of the respective dimension of the image. Can be greater
                than 1. For example, if the image width is 200 and `min_scale == 0.5`,
                then the width of the generated patch will be at least 100. If `min_scale == 1.5`,
                the width of the generated patch will be at least 300.
            max_scale (float, optional): The maximum size of a dimension of the patch
                as a fraction of the respective dimension of the image. Can be greater
                than 1. For example, if the image width is 200 and `max_scale == 1.0`,
                then the width of the generated patch will be at most 200. If `max_scale == 1.5`,
                the width of the generated patch will be at most 300. Must be greater than
                `min_scale`.
            scale_uniformly (bool, optional): If `True` and if `must_match == 'h_w'`,
                the patch height and width will be scaled uniformly, otherwise they will
                be scaled independently.
            min_aspect_ratio (float, optional): Determines the minimum aspect ratio
                for the generated patches.
            max_aspect_ratio (float, optional): Determines the maximum aspect ratio
                for the generated patches.
            patch_ymin (int, optional): `None` or the vertical coordinate of the top left
                corner of the generated patches. If this is not `None`, the position of the
                patches along the vertical axis is fixed. If this is `None`, then the
                vertical position of generated patches will be chosen randomly such that
                the overlap of a patch and the image along the vertical dimension is
                always maximal.
            patch_xmin (int, optional): `None` or the horizontal coordinate of the top left
                corner of the generated patches. If this is not `None`, the position of the
                patches along the horizontal axis is fixed. If this is `None`, then the
                horizontal position of generated patches will be chosen randomly such that
                the overlap of a patch and the image along the horizontal dimension is
                always maximal.
            patch_height (int, optional): `None` or the fixed height of the generated patches.
            patch_width (int, optional): `None` or the fixed width of the generated patches.
            patch_aspect_ratio (float, optional): `None` or the fixed aspect ratio of the
                generated patches.
        """

        if not (must_match in {'h_w', 'h_ar', 'w_ar'}):
            raise ValueError(
                "`must_match` must be either of 'h_w', 'h_ar' and 'w_ar'.")
        if min_scale >= max_scale:
            raise ValueError("It must be `min_scale < max_scale`.")
        if min_aspect_ratio >= max_aspect_ratio:
            raise ValueError(
                "It must be `min_aspect_ratio < max_aspect_ratio`.")
        if scale_uniformly and not ((patch_height is None) and
                                    (patch_width is None)):
            raise ValueError(
                "If `scale_uniformly == True`, `patch_height` and `patch_width` must both be `None`."
            )
        self.img_height = img_height
        self.img_width = img_width
        self.must_match = must_match
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_uniformly = scale_uniformly
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.patch_ymin = patch_ymin
        self.patch_xmin = patch_xmin
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_aspect_ratio = patch_aspect_ratio

    def __call__(self):
        """
        Returns:
            A 4-tuple `(ymin, xmin, height, width)` that represents the coordinates
            of the generated patch.
        """

        # Get the patch height and width.

        if self.must_match == 'h_w':  # Aspect is the dependent variable.
            if not self.scale_uniformly:
                # Get the height.
                if self.patch_height is None:
                    patch_height = int(
                        np.random.uniform(self.min_scale, self.max_scale) *
                        self.img_height)
                else:
                    patch_height = self.patch_height
                # Get the width.
                if self.patch_width is None:
                    patch_width = int(
                        np.random.uniform(self.min_scale, self.max_scale) *
                        self.img_width)
                else:
                    patch_width = self.patch_width
            else:
                scaling_factor = np.random.uniform(self.min_scale,
                                                   self.max_scale)
                patch_height = int(scaling_factor * self.img_height)
                patch_width = int(scaling_factor * self.img_width)

        elif self.must_match == 'h_ar':  # Width is the dependent variable.
            # Get the height.
            if self.patch_height is None:
                patch_height = int(
                    np.random.uniform(self.min_scale, self.max_scale) *
                    self.img_height)
            else:
                patch_height = self.patch_height
            # Get the aspect ratio.
            if self.patch_aspect_ratio is None:
                patch_aspect_ratio = np.random.uniform(self.min_aspect_ratio,
                                                       self.max_aspect_ratio)
            else:
                patch_aspect_ratio = self.patch_aspect_ratio
            # Get the width.
            patch_width = int(patch_height * patch_aspect_ratio)

        elif self.must_match == 'w_ar':  # Height is the dependent variable.
            # Get the width.
            if self.patch_width is None:
                patch_width = int(
                    np.random.uniform(self.min_scale, self.max_scale) *
                    self.img_width)
            else:
                patch_width = self.patch_width
            # Get the aspect ratio.
            if self.patch_aspect_ratio is None:
                patch_aspect_ratio = np.random.uniform(self.min_aspect_ratio,
                                                       self.max_aspect_ratio)
            else:
                patch_aspect_ratio = self.patch_aspect_ratio
            # Get the height.
            patch_height = int(patch_width / patch_aspect_ratio)

        # Get the top left corner coordinates of the patch.

        if self.patch_ymin is None:
            # Compute how much room we have along the vertical axis to place the patch.
            # A negative number here means that we want to sample a patch that is larger than the original image
            # in the vertical dimension, in which case the patch will be placed such that it fully contains the
            # image in the vertical dimension.
            y_range = self.img_height - patch_height
            # Select a random top left corner for the sample position from the possible positions.
            if y_range >= 0:
                patch_ymin = np.random.randint(
                    0, y_range + 1
                )  # There are y_range + 1 possible positions for the crop in the vertical dimension.
            else:
                patch_ymin = np.random.randint(
                    y_range, 1
                )  # The possible positions for the image on the background canvas in the vertical dimension.
        else:
            patch_ymin = self.patch_ymin

        if self.patch_xmin is None:
            # Compute how much room we have along the horizontal axis to place the patch.
            # A negative number here means that we want to sample a patch that is larger than the original image
            # in the horizontal dimension, in which case the patch will be placed such that it fully contains the
            # image in the horizontal dimension.
            x_range = self.img_width - patch_width
            # Select a random top left corner for the sample position from the possible positions.
            if x_range >= 0:
                patch_xmin = np.random.randint(
                    0, x_range + 1
                )  # There are x_range + 1 possible positions for the crop in the horizontal dimension.
            else:
                patch_xmin = np.random.randint(
                    x_range, 1
                )  # The possible positions for the image on the background canvas in the horizontal dimension.
        else:
            patch_xmin = self.patch_xmin

        return patch_ymin, patch_xmin, patch_height, patch_width


class CropPad:
    """
    Crops and/or pads an image deterministically.

    Depending on the given output patch size and the position (top left corner) relative
    to the input image, the image will be cropped and/or padded along one or both spatial
    dimensions.

    For example, if the output patch lies entirely within the input image, this will result
    in a regular crop. If the input image lies entirely within the output patch, this will
    result in the image being padded in every direction. All other cases are mixed cases
    where the image might be cropped in some directions and padded in others.

    The output patch can be arbitrary in both size and position as long as it overlaps
    with the input image.
    """

    def __init__(self,
                 patch_ymin,
                 patch_xmin,
                 patch_height,
                 patch_width,
                 clip_boxes=True,
                 box_filter=None,
                 background=(0, 0, 0),
                 labels_format={
                     'class_id': 0,
                     'xmin': 1,
                     'ymin': 2,
                     'xmax': 3,
                     'ymax': 4
                 }):
        """
        Arguments:
            patch_ymin (int, optional): The vertical coordinate of the top left corner of the output
                patch relative to the image coordinate system. Can be negative (i.e. lie outside the image)
                as long as the resulting patch still overlaps with the image.
            patch_ymin (int, optional): The horizontal coordinate of the top left corner of the output
                patch relative to the image coordinate system. Can be negative (i.e. lie outside the image)
                as long as the resulting patch still overlaps with the image.
            patch_height (int): The height of the patch to be sampled from the image. Can be greater
                than the height of the input image.
            patch_width (int): The width of the patch to be sampled from the image. Can be greater
                than the width of the input image.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the
                sampled patch.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the potential
                background pixels of the scaled images. In the case of single-channel images,
                the first element of `background` will be used as the background pixel value.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        """
        #  if (patch_height <= 0) or (patch_width <= 0):
        #    raise ValueError("Patch height and width must both be positive.")
        #  if (patch_ymin + patch_height < 0) or (patch_xmin + patch_width < 0):
        #    raise ValueError("A patch with the given coordinates cannot overlap with an input image.")
        if not (isinstance(box_filter, BoxFilter) or box_filter is None):
            raise ValueError(
                "`box_filter` must be either `None` or a `BoxFilter` object.")
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_ymin = patch_ymin
        self.patch_xmin = patch_xmin
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.background = background
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        if (self.patch_ymin > img_height) or (self.patch_xmin > img_width):
            raise ValueError(
                "The given patch doesn't overlap with the input image.")

        labels = np.copy(labels)

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        # Top left corner of the patch relative to the image coordinate system:
        patch_ymin = self.patch_ymin
        patch_xmin = self.patch_xmin

        # Create a canvas of the size of the patch we want to end up with.
        if image.ndim == 3:
            canvas = np.zeros(shape=(self.patch_height, self.patch_width, 3),
                              dtype=np.uint8)
            canvas[:, :] = self.background
        elif image.ndim == 2:
            canvas = np.zeros(shape=(self.patch_height, self.patch_width),
                              dtype=np.uint8)
            canvas[:, :] = self.background[0]

        # Perform the crop.
        if patch_ymin < 0 and patch_xmin < 0:  # Pad the image at the top and on the left.
            image_crop_height = min(
                img_height, self.patch_height + patch_ymin
            )  # The number of pixels of the image that will end up on the canvas in the vertical direction.
            image_crop_width = min(
                img_width, self.patch_width + patch_xmin
            )  # The number of pixels of the image that will end up on the canvas in the horizontal direction.
            canvas[-patch_ymin:-patch_ymin +
                   image_crop_height, -patch_xmin:-patch_xmin +
                   image_crop_width] = image[:image_crop_height, :
                                             image_crop_width]

        elif patch_ymin < 0 and patch_xmin >= 0:  # Pad the image at the top and crop it on the left.
            image_crop_height = min(
                img_height, self.patch_height + patch_ymin
            )  # The number of pixels of the image that will end up on the canvas in the vertical direction.
            image_crop_width = min(
                self.patch_width, img_width - patch_xmin
            )  # The number of pixels of the image that will end up on the canvas in the horizontal direction.
            canvas[-patch_ymin:-patch_ymin + image_crop_height, :
                   image_crop_width] = image[:image_crop_height, patch_xmin:
                                             patch_xmin + image_crop_width]

        elif patch_ymin >= 0 and patch_xmin < 0:  # Crop the image at the top and pad it on the left.
            image_crop_height = min(
                self.patch_height, img_height - patch_ymin
            )  # The number of pixels of the image that will end up on the canvas in the vertical direction.
            image_crop_width = min(
                img_width, self.patch_width + patch_xmin
            )  # The number of pixels of the image that will end up on the canvas in the horizontal direction.
            canvas[:image_crop_height, -patch_xmin:-patch_xmin +
                   image_crop_width] = image[patch_ymin:patch_ymin +
                                             image_crop_height, :
                                             image_crop_width]

        elif patch_ymin >= 0 and patch_xmin >= 0:  # Crop the image at the top and on the left.
            image_crop_height = min(
                self.patch_height, img_height - patch_ymin
            )  # The number of pixels of the image that will end up on the canvas in the vertical direction.
            image_crop_width = min(
                self.patch_width, img_width - patch_xmin
            )  # The number of pixels of the image that will end up on the canvas in the horizontal direction.
            canvas[:image_crop_height, :image_crop_width] = image[
                patch_ymin:patch_ymin +
                image_crop_height, patch_xmin:patch_xmin + image_crop_width]

        image = canvas

        if return_inverter:

            def inverter(labels):
                labels = np.copy(labels)
                labels[:, [ymin + 1, ymax + 1]] += patch_ymin
                labels[:, [xmin + 1, xmax + 1]] += patch_xmin
                return labels

        if not (labels is None):

            # Translate the box coordinates to the patch's coordinate system.
            labels[:, [ymin, ymax]] -= patch_ymin
            labels[:, [xmin, xmax]] -= patch_xmin

            # Compute all valid boxes for this patch.
            if not (self.box_filter is None):
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=self.patch_height,
                                         image_width=self.patch_width)

            if self.clip_boxes:
                labels[:, [ymin, ymax]] = np.clip(labels[:, [ymin, ymax]],
                                                  a_min=0,
                                                  a_max=self.patch_height - 1)
                labels[:, [xmin, xmax]] = np.clip(labels[:, [xmin, xmax]],
                                                  a_min=0,
                                                  a_max=self.patch_width - 1)

            if return_inverter:
                return image, labels, inverter
            else:
                return image, labels

        else:
            if return_inverter:
                return image, inverter
            else:
                return image


class Resize:
    """
    Resizes images to a specified height and width in pixels.
    """

    def __init__(self,
                 height,
                 width,
                 interpolation_mode=cv2.INTER_LINEAR,
                 box_filter=None,
                 labels_format={
                     'class_id': 0,
                     'xmin': 1,
                     'ymin': 2,
                     'xmax': 3,
                     'ymax': 4
                 }):
        """
        Arguments:
            height (int): The desired height of the output images in pixels.
            width (int): The desired width of the output images in pixels.
            interpolation_mode (int, optional): An integer that denotes a valid
                OpenCV interpolation mode. For example, integers 0 through 5 are
                valid interpolation modes.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        """
        if not (isinstance(box_filter, BoxFilter) or box_filter is None):
            raise ValueError(
                "`box_filter` must be either `None` or a `BoxFilter` object.")
        self.out_height = height
        self.out_width = width
        self.interpolation_mode = interpolation_mode
        self.box_filter = box_filter
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        image = cv2.resize(image,
                           dsize=(self.out_width, self.out_height),
                           interpolation=self.interpolation_mode)

        if return_inverter:

            def inverter(labels):
                labels = np.copy(labels)
                labels[:, [ymin + 1, ymax +
                           1]] = np.round(labels[:, [ymin + 1, ymax + 1]] *
                                          (img_height / self.out_height),
                                          decimals=0)
                labels[:, [xmin + 1, xmax +
                           1]] = np.round(labels[:, [xmin + 1, xmax + 1]] *
                                          (img_width / self.out_width),
                                          decimals=0)
                return labels

        if labels is None:
            if return_inverter:
                return image, inverter
            else:
                return image
        else:
            labels = np.copy(labels)
            labels[:, [ymin, ymax]] = np.round(labels[:, [ymin, ymax]] *
                                               (self.out_height / img_height),
                                               decimals=0)
            labels[:, [xmin, xmax]] = np.round(labels[:, [xmin, xmax]] *
                                               (self.out_width / img_width),
                                               decimals=0)
            # labels[:, [ymin, ymax]] = labels[:, [ymin, ymax]] * (self.out_height / img_height)
            # labels[:, [xmin, xmax]] = labels[:, [xmin, xmax]] * (self.out_width / img_width)
            if not (self.box_filter is None):
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=self.out_height,
                                         image_width=self.out_width)

            if return_inverter:
                return image, labels, inverter
            else:
                return image, labels


class ResizeRandomInterp:
    """
    Resizes images to a specified height and width in pixels using a radnomly
    selected interpolation mode.
    """

    def __init__(self,
                 height,
                 width,
                 interpolation_modes=[
                     cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC,
                     cv2.INTER_AREA, cv2.INTER_LANCZOS4
                 ],
                 box_filter=None,
                 labels_format={
                     'class_id': 0,
                     'xmin': 1,
                     'ymin': 2,
                     'xmax': 3,
                     'ymax': 4
                 }):
        """
        Arguments:
            height (int): The desired height of the output image in pixels.
            width (int): The desired width of the output image in pixels.
            interpolation_modes (list/tuple, optional): A list/tuple of integers
                that represent valid OpenCV interpolation modes. For example,
                integers 0 through 5 are valid interpolation modes.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        """
        if not (isinstance(interpolation_modes, (list, tuple))):
            raise ValueError("`interpolation_mode` must be a list or tuple.")
        self.height = height
        self.width = width
        self.interpolation_modes = interpolation_modes
        self.box_filter = box_filter
        self.labels_format = labels_format
        self.resize = Resize(height=self.height,
                             width=self.width,
                             box_filter=self.box_filter,
                             labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):
        self.resize.interpolation_mode = np.random.choice(
            self.interpolation_modes)
        self.resize.labels_format = self.labels_format
        return self.resize(image, labels, return_inverter)


def convert_coordinates(tensor, start_index, conversion, border_pixels='half'):
    """
    Convert coordinates for axis-aligned 2D boxes between two coordinate formats.

    Creates a copy of `tensor`, i.e. does not operate in place. Currently there are
    three supported coordinate formats that can be converted from and to each other:
        1) (xmin, xmax, ymin, ymax) - the 'minmax' format
        2) (xmin, ymin, xmax, ymax) - the 'corners' format
        2) (cx, cy, w, h) - the 'centroids' format

    Arguments:
        tensor (array): A Numpy nD array containing the four consecutive coordinates
            to be converted somewhere in the last axis.
        start_index (int): The index of the first coordinate in the last axis of `tensor`.
        conversion (str, optional): The conversion direction. Can be 'minmax2centroids',
            'centroids2minmax', 'corners2centroids', 'centroids2corners', 'minmax2corners',
            or 'corners2minmax'.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        A Numpy nD array, a copy of the input tensor with the converted coordinates
        in place of the original coordinates and the unaltered elements of the original
        tensor elsewhere.
    """
    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        tensor1[..., ind] = (tensor[..., ind] +
                             tensor[..., ind + 1]) / 2.0  # Set cx
        tensor1[..., ind + 1] = (tensor[..., ind + 2] +
                                 tensor[..., ind + 3]) / 2.0  # Set cy
        tensor1[..., ind +
                2] = tensor[..., ind + 1] - tensor[..., ind] + d  # Set w
        tensor1[..., ind +
                3] = tensor[..., ind + 3] - tensor[..., ind + 2] + d  # Set h
    elif conversion == 'centroids2minmax':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind +
                                                      2] / 2.0  # Set xmin
        tensor1[..., ind +
                1] = tensor[..., ind] + tensor[..., ind + 2] / 2.0  # Set xmax
        tensor1[..., ind +
                2] = tensor[..., ind +
                            1] - tensor[..., ind + 3] / 2.0  # Set ymin
        tensor1[..., ind +
                3] = tensor[..., ind +
                            1] + tensor[..., ind + 3] / 2.0  # Set ymax
    elif conversion == 'corners2centroids':
        tensor1[..., ind] = (tensor[..., ind] +
                             tensor[..., ind + 2]) / 2.0  # Set cx
        tensor1[..., ind + 1] = (tensor[..., ind + 1] +
                                 tensor[..., ind + 3]) / 2.0  # Set cy
        tensor1[..., ind +
                2] = tensor[..., ind + 2] - tensor[..., ind] + d  # Set w
        tensor1[..., ind +
                3] = tensor[..., ind + 3] - tensor[..., ind + 1] + d  # Set h
    elif conversion == 'centroids2corners':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind +
                                                      2] / 2.0  # Set xmin
        tensor1[..., ind +
                1] = tensor[..., ind +
                            1] - tensor[..., ind + 3] / 2.0  # Set ymin
        tensor1[..., ind +
                2] = tensor[..., ind] + tensor[..., ind + 2] / 2.0  # Set xmax
        tensor1[..., ind +
                3] = tensor[..., ind +
                            1] + tensor[..., ind + 3] / 2.0  # Set ymax
    elif (conversion == 'minmax2corners') or (conversion == 'corners2minmax'):
        tensor1[..., ind + 1] = tensor[..., ind + 2]
        tensor1[..., ind + 2] = tensor[..., ind + 1]
    else:
        raise ValueError(
            "Unexpected conversion value. Supported values are 'minmax2centroids', 'centroids2minmax', 'corners2centroids', 'centroids2corners', 'minmax2corners', and 'corners2minmax'."
        )

    return tensor1


def intersection_area_(boxes1,
                       boxes2,
                       coords='corners',
                       mode='outer_product',
                       border_pixels='half'):
    """
    The same as 'intersection_area()' but for internal use, i.e. without all the safety checks.
    """

    m = boxes1.shape[0]  # The number of boxes in `boxes1`
    n = boxes2.shape[0]  # The number of boxes in `boxes2`

    # Set the correct coordinate indices for the respective formats.
    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    elif coords == 'minmax':
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1  # If border pixels are supposed to belong to the bounding boxes, we have to add one pixel to any difference `xmax - xmin` or `ymax - ymin`.
    elif border_pixels == 'exclude':
        d = -1  # If border pixels are not supposed to belong to the bounding boxes, we have to subtract one pixel from any difference `xmax - xmin` or `ymax - ymin`.

    # Compute the intersection areas.

    if mode == 'outer_product':

        # For all possible box combinations, get the greater xmin and ymin values.
        # This is a tensor of shape (m,n,2).
        min_xy = np.maximum(
            np.tile(np.expand_dims(boxes1[:, [xmin, ymin]], axis=1),
                    reps=(1, n, 1)),
            np.tile(np.expand_dims(boxes2[:, [xmin, ymin]], axis=0),
                    reps=(m, 1, 1)))

        # For all possible box combinations, get the smaller xmax and ymax values.
        # This is a tensor of shape (m,n,2).
        max_xy = np.minimum(
            np.tile(np.expand_dims(boxes1[:, [xmax, ymax]], axis=1),
                    reps=(1, n, 1)),
            np.tile(np.expand_dims(boxes2[:, [xmax, ymax]], axis=0),
                    reps=(m, 1, 1)))

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:, :, 0] * side_lengths[:, :, 1]

    elif mode == 'element-wise':

        min_xy = np.maximum(boxes1[:, [xmin, ymin]], boxes2[:, [xmin, ymin]])
        max_xy = np.minimum(boxes1[:, [xmax, ymax]], boxes2[:, [xmax, ymax]])

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:, 0] * side_lengths[:, 1]


def iou(boxes1,
        boxes2,
        coords='centroids',
        mode='outer_product',
        border_pixels='half'):
    """
    Computes the intersection-over-union similarity (also known as Jaccard similarity)
    of two sets of axis-aligned 2D rectangular boxes.

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    In 'outer_product' mode, returns an `(m,n)` matrix with the IoUs for all possible
    combinations of the boxes in `boxes1` and `boxes2`.

    In 'element-wise' mode, `m` and `n` must be broadcast-compatible. Refer to the explanation
    of the `mode` argument for details.

    Arguments:
        boxes1 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(m, 4)` containing the coordinates for `m` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes2`.
        boxes2 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(n, 4)` containing the coordinates for `n` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes1`.
        coords (str, optional): The coordinate format in the input arrays. Can be either 'centroids' for the format
            `(cx, cy, w, h)`, 'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format
            `(xmin, ymin, xmax, ymax)`.
        mode (str, optional): Can be one of 'outer_product' and 'element-wise'. In 'outer_product' mode, returns an
            `(m,n)` matrix with the IoU overlaps for all possible combinations of the `m` boxes in `boxes1` with the
            `n` boxes in `boxes2`. In 'element-wise' mode, returns a 1D array and the shapes of `boxes1` and `boxes2`
            must be boadcast-compatible. If both `boxes1` and `boxes2` have `m` boxes, then this returns an array of
            length `m` where the i-th position contains the IoU overlap of `boxes1[i]` with `boxes2[i]`.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        A 1D or 2D Numpy array (refer to the `mode` argument for details) of dtype float containing values in [0,1],
        the Jaccard similarity of the boxes in `boxes1` and `boxes2`. 0 means there is no overlap between two given
        boxes, 1 means their coordinates are identical.
    """

    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2:
        raise ValueError(
            "boxes1 must have rank either 1 or 2, but has rank {}.".format(
                boxes1.ndim))
    if boxes2.ndim > 2:
        raise ValueError(
            "boxes2 must have rank either 1 or 2, but has rank {}.".format(
                boxes2.ndim))

    if boxes1.ndim == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4):
        raise ValueError(
            "All boxes must consist of 4 coordinates, but the boxes in `boxes1` and `boxes2` have {} and {} coordinates, respectively."
            .format(boxes1.shape[1], boxes2.shape[1]))
    if not mode in {'outer_product', 'element-wise'}:
        raise ValueError(
            "`mode` must be one of 'outer_product' and 'element-wise', but got '{}'."
            .format(mode))

    # Convert the coordinates if necessary.
    if coords == 'centroids':
        boxes1 = convert_coordinates(boxes1,
                                     start_index=0,
                                     conversion='centroids2corners')
        boxes2 = convert_coordinates(boxes2,
                                     start_index=0,
                                     conversion='centroids2corners')
        coords = 'corners'
    elif not (coords in {'minmax', 'corners'}):
        raise ValueError(
            "Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'."
        )

    # Compute the IoU.

    # Compute the interesection areas.

    intersection_areas = intersection_area_(boxes1,
                                            boxes2,
                                            coords=coords,
                                            mode=mode)

    m = boxes1.shape[0]  # The number of boxes in `boxes1`
    n = boxes2.shape[0]  # The number of boxes in `boxes2`

    # Compute the union areas.

    # Set the correct coordinate indices for the respective formats.
    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    elif coords == 'minmax':
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1  # If border pixels are supposed to belong to the bounding boxes, we have to add one pixel to any difference `xmax - xmin` or `ymax - ymin`.
    elif border_pixels == 'exclude':
        d = -1  # If border pixels are not supposed to belong to the bounding boxes, we have to subtract one pixel from any difference `xmax - xmin` or `ymax - ymin`.

    if mode == 'outer_product':

        boxes1_areas = np.tile(np.expand_dims(
            (boxes1[:, xmax] - boxes1[:, xmin] + d) *
            (boxes1[:, ymax] - boxes1[:, ymin] + d),
            axis=1),
                               reps=(1, n))
        boxes2_areas = np.tile(np.expand_dims(
            (boxes2[:, xmax] - boxes2[:, xmin] + d) *
            (boxes2[:, ymax] - boxes2[:, ymin] + d),
            axis=0),
                               reps=(m, 1))

    elif mode == 'element-wise':

        boxes1_areas = (boxes1[:, xmax] - boxes1[:, xmin] +
                        d) * (boxes1[:, ymax] - boxes1[:, ymin] + d)
        boxes2_areas = (boxes2[:, xmax] - boxes2[:, xmin] +
                        d) * (boxes2[:, ymax] - boxes2[:, ymin] + d)

    union_areas = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / union_areas
