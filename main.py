#%%
from preprocessing import preprocessing
from model import *
import tensorflow as tf
from validation import *
from keras import backend as K
tf.compat.v1.disable_eager_execution()

#%%
for case_nr in range(0,20):
    preprocessing(case_nr = case_nr ,slice_number_to_print = -1)

#%%

model = unet()

model.summary()
model_checkpoint = ModelCheckpoint(filepath = 'unet_kidneys.hdf5', monitor='loss',verbose=1, save_best_only=True)
#%%
for i in range(100):
    train_data, train_mask, test_data, test_mask = prep_data(i)
    model.fit(x = train_data, y = train_mask,epochs=2,callbacks=[model_checkpoint], batch_size=2)

#%%
train_data,train_mask,test_data,test_mask = prep_data(102)
results = model.predict(x = test_data, batch_size=2 ,verbose=1)
#%%
slice_nr = 120
display([test_data[slice_nr,:,:,:], test_mask[slice_nr,:,:,:], results[slice_nr,:,:,:]]) 

#%%
jacard = jaccard_distance_loss(K.variable(test_mask), K.variable(results.astype(np.float64))).eval(session = tf.compat.v1.keras.backend.get_session())

print('jaccard_distance_loss',jacard)
# %%
