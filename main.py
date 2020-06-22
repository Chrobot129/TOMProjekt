#%%
from preprocessing import preprocessing
from model import *
import tensorflow as tf
from validation import *
import tensorflow.keras.backend as K
from scipy.spatial.distance import directed_hausdorff

#%%
for case_nr in range(0,209):
    preprocessing(case_nr = case_nr ,slice_number_to_print = -1)

#%%
model = unet()


model.summary()
model_checkpoint = ModelCheckpoint(filepath = 'unet_kidneys.hdf5', monitor='loss',verbose=1, save_best_only=True)
plot = tf.keras.utils.plot_model(model, show_shapes=True, to_file="model.png", expand_nested = True )
#%%

for i in range(50):
    train_data, train_mask,x,y = prep_data(i)
    model.fit(x = train_data, y = train_mask,epochs=1,callbacks=[model_checkpoint], batch_size=2)

#%%
test_data, test_mask,x,y = prep_data(100)
results = model.predict(x = test_data, batch_size=2 ,verbose=1)
#%%
slice_nr = 200
display([test_data[slice_nr,:,:,:], test_mask[slice_nr,:,:,:], results[slice_nr,:,:,:]]) 

haus = directed_hausdorff(test_mask,results)

#%%
haus_list = []

for i in range(150, 201):
    test_data, test_mask,x,y = prep_data(i)
    results = model.predict(x = test_data, batch_size=2 ,verbose=1)
    for j in range(test_mask.shape[0]):

        haus = directed_hausdorff(test_mask[j,:,:,:],results[j,:,:,:])

        haus_list.append(haus)

arr = np.array(haus_list)
mean = np.mean(arr)
std = np.std(arr)




