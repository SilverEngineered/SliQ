import os
from sklearn.decomposition import PCA
import numpy as np
from sklearn import preprocessing
import random
from PIL import Image
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
def generate_pca_triplets(dataset, label_space=10, num_triplets=5000, testing=False):
    """
    Generates PCA triplet training examples from the specified dataset with specified qubit and label space sizes.
    :param dataset: String name of folder in datasets/[dataset] containing training and testing .npy files.
    :param num_qubits: Integer number of qubits 'q' defining 2^q PCA dimensions
    :param label_space: Integer number of labels to consider, filtering examples outside the label space.
    For MNIST, this corresponds to the lowest digit being filtered out.
    :return: The triplets used to train the Triplet Model - the list of n x 3 images, indices, and labels, where 'n'
    is the number of examples used. For these 3 return values, the first column is the anchor,
    the second the positive example, and the third the negative example. 'indices[j][1]' gives the index of the positive
    example of the jth triplet in the pre-triplet data, after filtering for the label space and performing PCA
    """

    if dataset == 'Landscape':
        return generate_landscape_triplets(num_triplets, testing)
    elif dataset == 'Aids':
        return generate_aids(num_triplets, testing)
    x, y = load_data(dataset, testing)
    x, y = filter_labels(x, y, label_space)
    x = scale_data(preprocessing.normalize(x))
    return generate_triplets(x, y, num_triplets)


def load_data(dataset, testing):
    """
    Loads training and testing data from datasets/[dataset]/x_train.npy and datasets/[dataset]/y_train.npy
    :param dataset: the folder within the 'datasets' folder containing the npy values.
    :return: The pair of loaded Numpy arrays for training features 'x' and labels 'y'
    """
    data_path = os.path.join("datasets", dataset)
    x_path = os.path.join(data_path, "x_train.npy")
    y_path = os.path.join(data_path, "y_train.npy")
    if testing:
        x_path = os.path.join(data_path, "x_test.npy")
        y_path = os.path.join(data_path, "y_test.npy")
    x = np.load(x_path)
    y = np.load(y_path)


    return x, y


def filter_labels(images, labels, label_space):
    """
    Filters examples if their label (integer) is not below the specified label space.
    :param images: Numpy array of image data, each row is a single image example.
    :param labels: Numpy array of example labels, corresponding 1:1 with the images.
    :param label_space: The # of labels to allow, resulting in keeping examples with labels [0, label_space - 1]
    :return: The list of remaining images and list of remaining labels.
    """
    filtered_images, filtered_labels = [], []
    for im, lab in zip(images, labels):
        if lab < label_space:
            filtered_images.append(im)
            filtered_labels.append(lab)
    return filtered_images, filtered_labels


def perform_pca(x, pca_dims=32):
    """
    Performs PCA on the given training example data with the specified # of dimensions. L2 normalizes data for
    PCA fit and transformation, and returns resulting features scaled to [0, 1] range.
    :param x: Numpy array containing rows of image data training examples.
    :param pca_dims: the number of dimensions (features) to reduce each example to via PCA.
    :return: Numpy array with [0, 1] scaled result of PCA dimensionality reduction.
    """
    pca = PCA(pca_dims)
    # Normalize image data so its pythagorean sum is 1
    pca.fit(preprocessing.normalize(x))
    return scale_data(pca.transform(preprocessing.normalize(x)))


def generate_triplets(x, y, size=5000):
    """
    Collects the specified number of triplet training examples from the given training data. Triplets
    have the first two elements in the same class (label) and the third in a separate class.
    :param x: Numpy array containing rows of training examples.
    :param y: Numpy array containing labels 1:1 with training examples.
    :param size: Number of triplets to collect.
    :return: 3 lists of 3-tiples, each corresponding to a triplet.
    The first is "triplets" containing the 3 examples of training data in the triplet.
    The second is "image_indices" giving the indices within the original data given for those examples.
    The third is "labels" giving the labels corresponding to each of the 3 examples such that:
    labels[n][0] == labels[n][1] != labels[n][2]
    """
    # List of 3-tuples picked from training examples.
    
    triplets = []
    image_indices = []
    labels = []
    for _ in range(size):
        index = p_index = n_index = int(np.floor(random.random() * len(x)))
        label = y[index]
        # Find a "positive" example, one in the SAME class as "index"
        while p_index == index or y[p_index] != label:
            p_index = int(np.floor(random.random() * len(x)))
        # Find a "negative" example, one in a DIFFERENT class as "index"
        while n_index == index or y[n_index] == label:
            n_index = int(np.floor(random.random() * len(x)))
        # Once 3 examples have been found, add examples, their indices, and their labels to output lists
        triplets.append((x[index], x[p_index], x[n_index]))
        image_indices.append((index, p_index, n_index))
        labels.append((y[index], y[p_index], y[n_index]))
    return triplets, image_indices, labels


def scale_data(data, scale=None, dtype=np.float32):
    """
    Scales every element in data linearly such that its minimum value is at the bottom of the specified scale,
    and the maximum value is at the top.
    :param data: Numpy array of data to scale.
    :param scale: A 2-element array, whose first value gives lower scale, second gives upper.
    :param dtype: Data type to transform scaled data to.
    :return: Data scaled as specified in the given type.
    """
    if scale is None:
        scale = [0, 1]
    min_data, max_data = [float(np.min(data)), float(np.max(data))]
    min_scale, max_scale = [float(scale[0]), float(scale[1])]
    data = ((max_scale - min_scale) * (data - min_data) / (max_data - min_data)) + min_scale
    return data.astype(dtype)



def generate_pca_triplets_batch(dataset, num_qubits, label_space=10, batch_size=2):
    x, y = load_data(dataset)
    x, y = filter_labels(x, y, label_space)
    triplets = []
    image_indices = []
    labels = []
    size = 4000
    for _ in range(size):
        index = p_index = n_index = int(np.floor(random.random() * len(x)))
        label = y[index]
        anchors_in = [index for i in range(batch_size)]
        pos_in = []
        negs_in = []
        while n_index == index or y[n_index] != label:
            n_index = int(np.floor(random.random() * len(x)))
        neg_label = y[n_index]
        # Find a "positive" example, one in the SAME class as "index"
        for i in range(batch_size):
            while p_index == index or y[p_index] != label:
                p_index = int(np.floor(random.random() * len(x)))
            pos_in.append(p_index)
        # Find a "negative" example, one in a DIFFERENT class as "index"
            while n_index == index or y[n_index] != neg_label:
                n_index = int(np.floor(random.random() * len(x)))
            negs_in.append(n_index)   
        anchors = [x[i] for i in anchors_in]
        negs = [x[i] for i in negs_in]
        pos = [x[i] for i in pos_in]

        triplets.append((anchors, negs, pos))
        image_indices.append((anchors_in, pos_in, negs_in))

        labels.append(([y[index] for i in range(batch_size)], [y[index] for i in range(batch_size)], [y[i] for i in negs_in]))
    return triplets, image_indices, labels
def get_color_density(image):
    num_bins = 8
    bin_threshold = 256 // num_bins
    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]
    red_bin = np.reshape(red // bin_threshold, [-1,])
    green_bin = np.reshape(green // bin_threshold, [-1])
    blue_bin = np.reshape(blue // bin_threshold, [-1])
    red_count = [0 for _ in range(num_bins)]
    blue_count = [0 for _ in range(num_bins)]
    green_count = [0 for _ in range(num_bins)]
    for r, g, b, in zip(red_bin, blue_bin, green_bin):
        red_count[r]+=1
        blue_count[g]+=1
        green_count[b]+=1
    return red_count, blue_count, green_count

def generate_landscape_triplets(num_triplets, testing=False):
    path = 'datasets/Landscapes'
    if testing:
        files = [f for f in listdir(path) if isfile(join(path, f))]
        files.sort()
        files = files[int(.8*len(files)):]
    else:
        files = [f for f in listdir(path) if isfile(join(path, f))]
        files.sort()
        files = files[:int(.8*len(files))]
    num_files = len(files)
    new_width, new_height = 50, 50
    image_triplets = []
    image_indecies = []
    for _ in range(num_triplets):
        indecies = []
        triplets = []
        images = []
        for i in range(3):
            index = int(np.floor(random.random() * num_files))
            im = Image.open('datasets/Landscapes/' + str(files[index]))
            width, height = im.size
            left = (width - new_width)//2
            top = (height - new_height)//2
            right = (width + new_width)//2
            bottom = (height + new_height)//2
            im = im.crop((left, top, right, bottom))
            r_bin, g_bin, b_bin = get_color_density(np.array(im))
            triplets.append(np.array(r_bin + g_bin + b_bin))
            indecies.append(index)
            images.append(np.reshape(np.array(im), [-1, ]))
        dist1 = np.sum(np.power(triplets[0]- triplets[1],2)) ** .5
        dist2 = np.sum(np.power(triplets[0]- triplets[2],2)) ** .5
        if dist1 < dist2:
            image_triplets.append((images[0], images[1], images[2]))
            image_indecies.append((indecies[0], indecies[1], indecies[2]))
        else:
            image_triplets.append((images[0], images[1], images[2]))
            image_indecies.append((indecies[0], indecies[2], indecies[1]))
    return image_triplets, image_indecies, None

def generate_aids(num_triplets=5000, testing=False):
    path = 'datasets/Aids'
    triplet_labels = []
    with open(os.path.join(path, 'aids_conc_may04.txt')) as f:
        lines = f.read()
    lines = [tuple(i.replace(' ', '').split(',')) for i in lines.split('\n') if tuple(i.replace(' ', '').split(','))[0].isnumeric()][1:]
    labels = [(int(i[0]), i[1]) for i in lines]
        
    data = {}
    with open(os.path.join(path, 'aids_ec50_may04.txt')) as f:
        lines = f.read()

    for line in lines.split('\n')[1:]:
        if len(line.split(',')) == 7:
            NSC,Log10HiConc,ConcUnit,Flag,Log10EC50,NumExp,StdDev = line.split(',')
            Flag = 0 if Flag == '>' else 1
            data[int(NSC)] = [Log10HiConc, Flag, Log10EC50, NumExp]
    indecies = []
    triplets = []
    images = []
    label_set = ['CI', 'CM', 'CA']
    for l in tqdm(range(1500)):
        index = None
        label = None
        if l < num_triplets / 3:
            needed_class = 'CI'
        elif l < (2 * num_triplets) / 3:
            needed_class = 'CM'
        else:
            needed_class = 'CA'
        while index not in data or label != needed_class:
            anchor = labels[int(np.floor(random.random() * len(labels)))]
            index, label = anchor[0], anchor[1]
        n_index = None
        p_index = None
        p_label = None
        n_label = label
        while p_label != label or not p_index in data:
            p = labels[int(np.floor(random.random() * len(labels)))]
            p_index, p_label = p[0], p[1]

        bad_label = random.random() > .5
        if label == 'CM' and bad_label:
            bad_label = 'CA'
        elif label == 'CM':
            bad_label = 'CI'

        if label == 'CA' and bad_label:
            bad_label = 'CI'
        elif label == 'CA':
            bad_label = 'CM'
        # Find a "negative" example, one in a DIFFERENT class as "index"
        while n_label == label or not n_index in data or n_label == bad_label:
            n = labels[int(np.floor(random.random() * len(labels)))]
            n_index, n_label = n[0], n[1]

        indecies.append((index, p_index, n_index))
        triplet_labels.append((label, p_label, n_label))
        triplets.append((data[index], data[p_index], data[n_index]))
    test_range = [i for i in range(1500) if i < 200 or (i < 700 and i > 500) or (i > 1300)]
    if testing:
        triplets = [triplets[i][0] for i in test_range]
        indecies = [indecies[i][0] for i in test_range]
        triplet_labels = [triplet_labels[i][0] for i in test_range]
    else:
        triplets = [triplets[i][0] for i in range(1500) if i not in test_range]
        indecies = [indecies[i][0] for i in range(1500) if i not in test_range]
        triplet_labels = [triplet_labels[i][0] for i in range(1500) if i not in test_range]
    return triplets, indecies, triplet_labels