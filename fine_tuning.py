from keras.applications import resnet50

conv_base = resnet50.ResNet50(weights='imagenet', include_top=False)
#    include_top=False)

conv_base.summary()
from keras import models
from keras import layers
model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.layers[0].trainable = False #don't train conv
model.summary()
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



from keras.preprocessing.image import ImageDataGenerator

train_dir = 'C:/Users/samuelmao/Desktop/wardrobe_detection/image/wardrobe_nonwardrobe/train'
validation_dir = 'C:/Users/samuelmao/Desktop/wardrobe_detection/image/wardrobe_nonwardrobe/test'

train_datagen= ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255) # define a generator

train_generator = train_datagen.flow_from_directory(train_dir, # the training image file path
                                                 target_size=(224,224), # resnet takes input of size 244 by 244 (224 perhaps?)
                                                 batch_size=20, # batch size for loading by generator
                                                 class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir, # the training image file path
                                                 target_size=(224,224), # resnet takes input of size 244 by 244
                                                 batch_size=20, # batch size for loading by generator
                                                 class_mode='binary')

step_size_train=train_generator.n//train_generator.batch_size
step_size_test=validation_generator.n//validation_generator.batch_size

from keras.callbacks import EarlyStopping, ModelCheckpoint
early = EarlyStopping(monitor="val_loss", mode="min", patience=2)
checkpoint = ModelCheckpoint(filepath='fine_tune_dataaug.hdf5', monitor='val_fmeasure', save_best_only=True)

# model.compile(loss='binary_crossentropy',
#     optimizer=optimizers.RMSprop(lr=1e-4),
#     metrics=[fmeasure])

model.compile(loss='binary_crossentropy',
    optimizer=optimizers.Adam(),
    metrics=[fmeasure])

history = model.fit_generator(
    train_generator,
    steps_per_epoch = step_size_train,
    epochs = 30,
    validation_data = validation_generator,
    validation_steps = step_size_test,
    callbacks=[early, checkpoint]
)
