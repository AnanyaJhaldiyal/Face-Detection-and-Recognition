import os
import imageio
import matplotlib.pyplot as plt
new_image_size = (150,150,3)
# set the directory containing the images
images_dir = './Headshots'
current_id = 0
# for storing the foldername: label,
label_ids = {}
# for storing the images data and labels
images = []
labels = []
for root, _, files in os.walk(images_dir):
    for file in files:
        if file.endswith(('png','jpg','jpeg')):
            # path of the image
            path = os.path.join(root, file)

            # get the label name
            label = os.path.basename(root).replace(
                " ", ".").lower()

            # add the label (key) and its number (value)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            # save the target value
            labels.append(current_id-1)
            # load the image, resize and flatten it
            image = imread(path)
            image = resize(image, new_image_size)
            images.append(image.flatten())

            # show the image
            plt.imshow(image, cmap=plt.cm.gray_r,interpolation='nearest')
            plt.show()
print(label_ids)
# save the labels for each fruit as a list
categories = list(label_ids.keys())
pickle.dump(categories, open("faces_labels.pk", "wb" ))
df = pd.DataFrame(np.array(images))
df['Target'] = np.array(labels)
df
