from keras.applications import VGG16

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
#    include_top=False)

from keras.preprocessing.image import ImageDataGenerator

train_dir = './image/wardrobe_nonwardrobe/train'
validation_dir = './image/wardrobe_nonwardrobe/test'
testing_dir = './image/wardrobe_nonwardrobe/testing'
#\\tsclient\C\Users\samuelmao\Desktop\wardrobe_detection
# train_datagen= ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)
data_datagen = ImageDataGenerator(rescale=1./255) # define a generator
batch_size=20
train_datagen = ImageDataGenerator(rescale=1./255) # define a generator
import numpy as np
test_datagen = ImageDataGenerator(rescale=1./255) # define a generator
def extract_features(directory,sample_count):
    features=np.zeros(shape=(sample_count,4,4,512))
    labels=np.zeros(shape=(sample_count))
    generator = data_datagen.flow_from_directory(
    directory,
    target_size=(150,150),
    batch_size=batch_size,
    class_mode='binary'
    )
    i=0
    for inputs_batch, labels_batch in generator:
        features_batch=conv_base.predict(inputs_batch)
        features[i*batch_size:(i+1)*batch_size] = features_batch
        labels[i*batch_size:(i+1)*batch_size]=labels_batch
        i+=1
        print(i,str(directory))
        if i * batch_size >= sample_count:
            break
    return features, labels
train_sample=len(data_datagen.flow_from_directory(train_dir).filenames)

val_sample=len(data_datagen.flow_from_directory(validation_dir).filenames)

testing_sample = len(data_datagen.flow_from_directory(testing_dir).filenames)

train_features, train_labels = extract_features(train_dir,train_sample)
print("extracted training feature")
validation_features, validation_labels = extract_features(validation_dir,val_sample)

testing_features, testing_labels = extract_features(testing_dir,testing_sample)

train_features = np.reshape(train_features,(train_sample,4*4*512))
validation_features = np.reshape(validation_features,(val_sample,4*4*512))
testing_features = np.reshape(testing_features,(testing_sample,4*4*512))

from keras import models
from keras import layers
model1 = models.Sequential()
model1.add(layers.Dense(256, activation='relu',input_dim=4*4*512))
model1.add(layers.Dropout(0.5))
model1.add(layers.Dense(1, activation='sigmoid'))
model1.summary()
print("start modeling........")
from keras import optimizers
import keras.backend as K
def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def fbeta_score(y_true, y_pred, beta=1):

    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    '''Calculates the f-measure, the harmonic mean of precision and recall.
    '''
    return fbeta_score(y_true, y_pred, beta=1)





from keras.callbacks import EarlyStopping, ModelCheckpoint
early = EarlyStopping(monitor="val_loss", mode="min", patience=3)
checkpoint = ModelCheckpoint(filepath='vgg16.hdf5', monitor='val_fmeasure', save_best_only=True)


from keras import models

model1.compile(loss='binary_crossentropy',
    optimizer=optimizers.Adam(),
    metrics=[fmeasure])



history = model1.fit(train_features, train_labels,
epochs = 30,
batch_size=20,
validation_data = (validation_features,validation_labels),
callbacks=[early, checkpoint])

model1.save('vgg_saved.h5')

### predict and visualize outcome
from keras.preprocessing import image
from keras.models import load_model
model = load_model('vgg_saved.h5', custom_objects={'fmeasure':fmeasure})
model.compile(loss='binary_crossentropy',
    optimizer=optimizers.Adam(),
    metrics=[fmeasure])

img_arrays = np.expand_dims(testing_features, axis=0)
preds = [model.predict_classes(x) for x in img_arrays]
preds


import matplotlib.pyplot as plt
import cv2

f, ax = plt.subplots(6, 6, figsize = (15, 15))
for i in range(0,36):
    imgBGR = cv2.imread(files[i])
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

    # a if condition else b
    predicted_class = "wardrobe" if preds[0][i]==1 else "non_wardrobe"

    ax[i//6, i%6].imshow(imgRGB)
    ax[i//6, i%6].axis('off')
    ax[i//6, i%6].set_title("Predicted:{}".format(predicted_class))
plt.show()
f.savefig('C:/Users/samuelmao/Desktop/prediction')
##
