# from Deep learning With Python
#best validation f1: 0.946
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

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


model.compile(loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=[fmeasure])

from keras.preprocessing.image import ImageDataGenerator

train_dir = 'C:/Users/samuelmao/Desktop/wardrobe_detection/image/wardrobe_nonwardrobe/train'
validation_dir = 'C:/Users/samuelmao/Desktop/wardrobe_detection/image/wardrobe_nonwardrobe/test'

train_datagen = ImageDataGenerator(rescale=1./255) # define a generator
test_datagen = ImageDataGenerator(rescale=1./255) # define a generator

train_generator = train_datagen.flow_from_directory(train_dir, # the training image file path
                                                 target_size=(150,150), # resnet takes input of size 244 by 244
                                                 batch_size=20, # batch size for loading by generator
                                                 class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir, # the training image file path
                                                 target_size=(150,150), # resnet takes input of size 244 by 244
                                                 batch_size=20, # batch size for loading by generator
                                                 class_mode='binary')

step_size_train=train_generator.n//train_generator.batch_size
step_size_test=validation_generator.n//validation_generator.batch_size

from keras.callbacks import EarlyStopping, ModelCheckpoint
early = EarlyStopping(monitor="val_loss", mode="min", patience=2)
checkpoint = ModelCheckpoint(filepath='best.hdf5', monitor='val_loss', save_best_only=True)

history = model.fit_generator(
    train_generator,
    steps_per_epoch = step_size_train,
    epochs = 30,
    validation_data = validation_generator,
    validation_steps = step_size_test,
    callbacks=[early, checkpoint]
)

model.save('simple_model.h5')

#plot model
import matplotlib.pyplot as plt
f1 = history.history['fmeasure']
val_f1 = history.history['val_fmeasure']
epochs = range(1, len(f1)+1)
plt.plot(epochs,f1,'bo',label='f1_training')
plt.plot(epochs, val_f1,'b', label = 'f1_validation')
plt.legend()

#load model
from keras.models import load_model
best = load_model('best.hdf5',custom_objects={'fmeasure':fmeasure}) #this loads self-defined metrics
