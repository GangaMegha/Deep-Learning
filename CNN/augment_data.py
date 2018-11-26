import Augmentor


p = Augmentor.Pipeline("../Dataset/class_0/")
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
p.flip_top_bottom(probability=0.005)

p.sample(10000)
p = Augmentor.Pipeline("../Dataset/class_1/")
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
p.flip_top_bottom(probability=0.005)

p.sample(10000)


p = Augmentor.Pipeline("../Dataset/class_2/")
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
p.flip_top_bottom(probability=0.005)

p.sample(10000)


p = Augmentor.Pipeline("../Dataset/class_3/")
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
p.flip_top_bottom(probability=0.005)

p.sample(10000)


p = Augmentor.Pipeline("../Dataset/class_4/")
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
p.flip_top_bottom(probability=0.005)

p.sample(10000)


p = Augmentor.Pipeline("../Dataset/class_5/")
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
p.flip_top_bottom(probability=0.005)

p.sample(10000)


p = Augmentor.Pipeline("../Dataset/class_6/")
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
p.flip_top_bottom(probability=0.005)

p.sample(10000)


p = Augmentor.Pipeline("../Dataset/class_7/")
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
p.flip_top_bottom(probability=0.005)

p.sample(10000)

p = Augmentor.Pipeline("../Dataset/class_8/")
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
p.flip_top_bottom(probability=0.005)

p.sample(10000)


p = Augmentor.Pipeline("../Dataset/class_9/")
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
p.flip_top_bottom(probability=0.005)

p.sample(10000)