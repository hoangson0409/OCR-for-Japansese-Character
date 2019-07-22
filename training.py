from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from Image_Generator import TextImageGenerator
from Model import get_Model
import time
from callbacks import TrainCheck
from parameter import *
K.set_learning_phase(0)

# # Model description and training

model = get_Model(training=True)

try:
    model.load_weights('./model_OCR/LSTM+BN5--08--1.671.hdf5')
    print("...Previous weight data...")
except:
    print("...New weight data...")
    pass
print(model.summary())

img_train_path = './data_japan/train/'
json_train_path = './data_japan/labels/train.json'
tiger_train = TextImageGenerator( img_train_path, json_train_path, img_w, img_h, batch_size,'train', downsample_factor)
#tiger_train.build_data()

img_val_path = './data_japan/test/'
json_val_path = './data_japan/labels/test.json'
tiger_val = TextImageGenerator(img_val_path, json_val_path, img_w, img_h, val_batch_size,'val', downsample_factor)
#tiger_val.build_data()
print('>>>>>>>>>>>>>>>>>',val_batch_size)

ada = Adam()


early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=1)
checkpoint = ModelCheckpoint(filepath='./model_OCR/' +'LSTM+BN5--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='loss', verbose=1, mode='min', period=1)
tensorboard = TensorBoard(log_dir="logs_OCR/LSTM+BN5{}".format(time.time()),
                              batch_size=batch_size, write_images=True)
train_check = TrainCheck()
# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)

# captures output of softmax so we can decode the output during visualization
model.fit_generator(generator=tiger_train.next_batch(),
                    steps_per_epoch=int(tiger_train.n / batch_size),
                    epochs=30,
                    callbacks=[checkpoint,train_check,tensorboard],
                    validation_data=tiger_val.next_batch(),
                    validation_steps=int(tiger_val.n / val_batch_size),initial_epoch = 8 )
