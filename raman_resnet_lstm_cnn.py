import csv
import itertools
import math
import pprint
import tkinter
from random import random
from PIL import Image, ImageDraw
import numpy as np
from keras import Input
from keras.applications.resnet import ResNet, ResNet50, ResNet101
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.layers import LSTM, Dense, BatchNormalization, Dropout, Flatten, LeakyReLU
from keras.models import Sequential, Model
from keras.optimizer_v2.adam import Adam
from keras.optimizer_v2.gradient_descent import SGD
from keras.optimizer_v2.rmsprop import RMSProp
from keras.regularizers import L2
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.models import load_model
from keras.optimizer_v2.nadam import Nadam
import itertools

datasets = ["40NTNC_40TNC_Nucleus_FP.csv"]
testsets = ["20mix_with_labels.csv", "30mix_with_labels.csv"]

window_size = 364
image_save = False

windowed_x_all = []
windowed_y_all = []

Model()


def generate_images(dataset, y=None, aug=False, aug_size=10, aug_ratio=0.2, tag=""):
    images = []
    img_targets = []
    print(tag, "Augmentation: ({0},{1},{2})".format(aug, aug_size, aug_ratio))
    print(tag, "Generating {0} images...".format(dataset.shape[0]))
    for i in range(0, dataset.shape[0]):
        image1 = generate_single_image_orig(dataset, i, y, 0, tag)
        images.append(np.array(image1))
        if y is not None:
            img_targets.append(y[i])
        if image_save:
            image1.save("./genimages/{0}.png".format(i))
        if aug:
            augs = generate_augmented_images_orig(dataset, i, aug_size, aug_ratio)
            auidx = 0
            for ai in augs:
                images.append(np.array(ai))
                if y is not None:
                    img_targets.append(y[i])
                if image_save:
                    ai.save("./genimages/{0}_{1}.png".format(i, auidx))
                auidx += 1

    if y is not None:
        return images, img_targets
    else:
        return images


def generate_augmented_images(dataset, i, n, ratio, xshift=0):
    aimages = []
    m = max(dataset[i]) * max(dataset[i])
    for a in range(0, n):
        image1 = Image.new("RGB", (364, 256), (255, 255, 255))
        draw = ImageDraw.Draw(image1)
        pos = 0
        grow = 0
        prev = dataset[i][0]
        for x in dataset[i]:
            x = x * x
            if x > 0:
                s = 0
            else:
                s = 0
            grow = (s, 0, 0)
            pos_final = min(364, max(0, pos + int(xshift * random() if random() * ratio > ratio / 2 else 0)
                                     - int(xshift * random() if random() * ratio > ratio / 2 else 0)))
            draw.rectangle([pos, 256, pos_final,
                            256 - 256 * ((abs(x) / m) * (1 + ratio * random()))], grow)
            pos += 1
            prev = x
        aimages.append(image1)
    return aimages


def generate_augmented_images_orig(dataset, i, n, ratio):
    aimages = []
    m = max(dataset[i])
    for a in range(0, n):
        image1 = Image.new("RGB", (364, 256), (255, 255, 255))
        draw = ImageDraw.Draw(image1)
        pos = 0
        grow = 0
        prev = dataset[i][0]
        for x in dataset[i]:
            if x > 0:
                s = 0
            else:
                s = 0
            grow = (s, 0, 0)
            draw.rectangle([pos, 256, pos,
                            256 - 256 * ((abs(x) / m) * (1 + ratio * random()))], grow)
            pos += 1
            prev = x
        aimages.append(image1)
    return aimages


def generate_single_image_orig(dataset, i, y, ratio, tag=""):
    m = max(dataset[i])
    # print(tag,"Generating image {0}...".format(i))
    image1 = Image.new("RGB", (364, 256), (255, 255, 255))
    draw = ImageDraw.Draw(image1)
    pos = 0
    grow = 0
    prev = dataset[i][0]
    for x in dataset[i]:
        if x > 0:
            s = 0
        else:
            s = 0
        grow = (s, 0, 0)
        draw.line([pos, 256, pos,
                   256 - 256 * ((abs(x) / m) * (1 + ratio * random()))], grow)
        pos += 1
        prev = x
    return image1


def generate_single_image(dataset, i, y, ratio, tag=""):
    m = max(dataset[i]) * max(dataset[i])
    # print(tag,"Generating image {0}...".format(i))
    image1 = Image.new("RGB", (364, 256), (255, 255, 255))
    draw = ImageDraw.Draw(image1)
    pos = 0
    grow = 0
    prev = dataset[i][0]
    for x in dataset[i]:
        x = x * x
        if x > 0:
            s = 0
        else:
            s = 0
        grow = (s, 0, 0)
        draw.line([pos, 256, pos,
                   256 - 256 * ((abs(x) / m) * (1 + ratio * random()))], grow)
        pos += 1
        prev = x
    return image1


def generate_windows_full(samples):
    target = samples[len(samples) - 1]
    row_samples = [float(x) for x in samples[:len(samples) - 1]]

    local_windows = [row_samples]
    local_target = [1 if target == 'TNC' else 0]

    return local_windows, local_target


def generate_windows_cid(samples):
    target = samples[len(samples) - 1]
    row_samples = [float(x) for x in samples[:len(samples) - 1]]

    local_windows = [row_samples]
    local_target = [str(target).split('.')[0]]

    return local_windows, local_target


def generate_windows_unlabeled(row_samples):
    row_floats = [float(x) for x in row_samples]
    local_windows = [row_floats]
    return local_windows


def create_model(inputshape, nclasses, optimizer, fine_tune=0):
    conv_base = ResNet101(include_top=False,
                          weights='imagenet',
                          input_shape=inputshape)
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in conv_base.layers:
            layer.trainable = False
    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = BatchNormalization()(top_model)
    top_model = Dense(512, activation=LeakyReLU(), kernel_initializer='glorot_normal', kernel_regularizer=L2())(
        top_model)
    top_model = Dropout(0.2)(top_model)
    top_model = Dense(256, activation=LeakyReLU(), kernel_initializer='glorot_normal', kernel_regularizer=L2())(
        top_model)
    top_model = Dropout(0.2)(top_model)
    top_model = Dense(128, activation=LeakyReLU(), kernel_initializer='glorot_normal', kernel_regularizer=L2())(
        top_model)
    top_model = Dropout(0.2)(top_model)
    top_model = Dense(64, activation=LeakyReLU(), kernel_initializer='glorot_normal', kernel_regularizer=L2())(
        top_model)
    top_model = Dropout(0.2)(top_model)
    top_model = Dense(32, activation=LeakyReLU(), kernel_initializer='glorot_normal', kernel_regularizer=L2())(
        top_model)
    top_model = Dropout(0.2)(top_model)
    output_layer = Dense(nclasses, activation='sigmoid')(top_model)
    model = Model(inputs=conv_base.input, outputs=output_layer)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


train_enabled = True
test_enabled = True

if train_enabled:
    for dst in datasets:
        print("File:" + dst)
        freqs = []
        with open("data/raman/" + dst) as csvfile:
            reader = csv.reader(csvfile)
            skipline = 0
            rowcount = 0
            for row in reader:
                if skipline == 0:
                    freqs = row
                    skipline += 1
                    continue
                rowcount += 1
                if rowcount % 1 == 0:
                    windows, targets = generate_windows_full(row)
                    for w in windows:
                        windowed_x_all.append(w)
                    for t in targets:
                        windowed_y_all.append(t)

    windowed_x_all = np.array(windowed_x_all, dtype=np.float32)

    x_trva, x_test, y_trva, y_test = train_test_split(windowed_x_all, windowed_y_all, test_size=0.30, shuffle=True)

    x_test, y_test = generate_images(x_test, y_test, False, tag="TEST")
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_train, x_val, y_train, y_val = train_test_split(x_trva, y_trva, test_size=0.20, shuffle=True)

    x_train, y_train = generate_images(x_train, y_train, True, aug_size=10, aug_ratio=0.15, tag="TRAIN")
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_val, y_val = generate_images(x_val, y_val, False, aug_size=10, aug_ratio=0.15, tag="VALIDATION")
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)

    input_shape = (256, 364, 3)
    optim = RMSProp(learning_rate=0.00001)
    # optim = Nadam()
    n_classes = 1

    BATCH_SIZE = 4
    n_steps = len(x_train) // BATCH_SIZE
    n_val_steps = len(x_val) // BATCH_SIZE
    n_epochs = 50

    neural_model = create_model(input_shape, n_classes, optim, fine_tune=3)

    best_previous_model = 0
    reports = {}


    def test_model(p_neural_model, p_test_windowed_y_all, p_x_test, p_dataset_name, p_epoch):
        l_y_predict = p_neural_model.predict(p_x_test, verbose=0)
        l_y_pred = list(itertools.chain.from_iterable(l_y_predict))
        l_decisions = [1 if x > 0.5 else 0 for x in l_y_pred]
        print("[{0}]".format(p_epoch), "({0})".format(p_dataset_name) + "Decisions:", l_decisions)
        print("[{0}]".format(p_epoch), "({0})".format(p_dataset_name) + "Cell ids:", p_test_windowed_y_all)
        print(len(l_y_pred))
        print("[{0}]".format(p_epoch), "({0})".format(p_dataset_name) + "Spectra Level Healthy:",
              100 * (len(l_y_pred) - sum(l_decisions)) / len(l_y_pred))
        print("[{0}]".format(p_epoch), "({0})".format(p_dataset_name) + "Spectra Level Tumoral:",
              100 * sum(l_decisions) / len(l_y_pred))
        if sum(l_decisions) / len(l_y_pred) < 0.55:
            print("[{0}]".format(p_epoch), "Spectra Tumoral Level under threshold, going to discard trial.")
            return True
        print("[{0}]".format(p_epoch), 20 * "-", " AGGREGAZIONE PER CELLULA File:{0}".format(p_dataset_name), 20 * "-")
        l_cell_id_max = max(np.array(p_test_windowed_y_all, dtype=int))
        l_decs = np.array(l_decisions)
        l_ids = np.array(p_test_windowed_y_all, dtype=int)
        for element in itertools.product([0.6, 0.7, 0.8], [0.2, 0.3, 0.4]):
            l_cell_decisions = []
            print("[{0}]".format(p_epoch), " Element:", element)
            for i in range(1, l_cell_id_max + 1):
                l_m = np.ma.masked_where(l_ids != i, l_decs)
                l_mc = l_m.compressed()
                if sum(l_mc) / len(l_mc) >= element[0]:
                    l_cell_decisions.append(1)
                    print("[{0}] [{1}]".format(p_epoch, element), "Processing cell:" + str(i),
                          "Spectra classifications:", l_mc, "Tumoral %:",
                          sum(l_mc) / len(l_mc), "Tumoral cell")
                elif sum(l_mc) / len(l_mc) <= element[1]:
                    l_cell_decisions.append(0)
                    print("[{0}] [{1}]".format(p_epoch, element), "Processing cell:" + str(i),
                          "Spectra classifications:", l_mc, "Tumoral %:",
                          sum(l_mc) / len(l_mc), "Healthy cell")
                else:
                    print("[{0}] [{1}]".format(p_epoch, element), "Processing cell:" + str(i),
                          "Spectra classifications:", l_mc, "Tumoral %:",
                          sum(l_mc) / len(l_mc), "Discarded cell")
            print("[{0}] [{1}]".format(p_epoch, element), "Included cells:", len(l_cell_decisions))
            print("[{0}] [{1}]".format(p_epoch, element), "Cell Level Healthy:",
                  100 * len([x for x in l_cell_decisions if x == 0]) / len(l_cell_decisions))
            print("[{0}] [{1}]".format(p_epoch, element), "Cell Level Tumoral:",
                  100 * len([x for x in l_cell_decisions if x == 1]) / len(l_cell_decisions))
        return False


    class TestCallback(Callback):
        def on_epoch_end(self, epoch, epoch_logs):
            global best_previous_model, neural_model, x_test
            test_y_predict = neural_model.predict(x_test, verbose=1)
            test_y_pred = list(itertools.chain.from_iterable(test_y_predict))
            report = classification_report(y_test, [1 if x > 0.5 else 0 for x in test_y_pred], output_dict=True)
            f1ma = report['macro avg']['f1-score']
            reports[str(epoch)] = f1ma
            if best_previous_model < f1ma:
                print(20 * "-", "New model {0} is better than previous {1}, dumping and overwriting:"
                      .format(f1ma, best_previous_model),
                      'model_' + str(epoch) + ".h5")
                neural_model.save('models/model_' + str(epoch) + ".h5")
                neural_model.save('models/best_model.h5', overwrite=True)
                best_previous_model = f1ma
            else:
                print("[{0}]".format(epoch),4 * "-", "Current model is worst then best_model.")

            pprint.pprint(report)
            print("[{0}]".format(epoch), " -- END OF EPOCH", 20 * '-')

            if test_enabled:
                for test_set in testsets:
                    test_windowed_x_all = []
                    test_windowed_y_all = []
                    for t in [test_set]:
                        print("[{0}]".format(epoch), "File:" + test_set)
                        freqs = []
                        with open("data/raman/" + test_set) as csvfile:
                            reader = csv.reader(csvfile)
                            skipline = 0
                            rowcount = 0
                            for row in reader:
                                if skipline == 0:
                                    freqs = row
                                    skipline += 1
                                    continue
                                rowcount += 1
                                if rowcount % 1 == 0:
                                    windows, cellids = generate_windows_cid(row)
                                    for w in windows:
                                        test_windowed_x_all.append(w)
                                    for cid in cellids:
                                        test_windowed_y_all.append(cid)

                    test_windowed_x_all = np.array(test_windowed_x_all, dtype=np.float32)
                    ed_x_test, labels = generate_images(test_windowed_x_all, test_windowed_y_all, aug=False)
                    ed_x_test = np.array(ed_x_test)

                    print("[{0}]".format(epoch), 10 * '-', "TESTING {0}".format(test_set))
                    if test_model(neural_model, test_windowed_y_all, ed_x_test, test_set, epoch):
                        print("[{0}]".format(epoch), "({0})".format(test_set) + "Stopped useless trial.")
                        return


    history = neural_model.fit(x_train, y_train,
                                   batch_size=BATCH_SIZE,
                                   epochs=n_epochs,
                                   validation_data=(x_val, y_val),
                                   steps_per_epoch=n_steps,
                                   validation_steps=n_val_steps,
                                   callbacks=[TestCallback()],
                                   verbose=1)

    print(history)
