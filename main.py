#%%
from preprocessing import preprocessing
from model import *

#%%
for case_nr in range(0,20):
    preprocessing(case_nr = case_nr ,slice_number_to_print = -1)

# %%


def prep_data(case_nr):
    train_data, train_mask  = get_data(case_nr)
    test_data, test_mask = get_data(case_nr+1)

    train_data = np.expand_dims(train_data, axis=3)
    train_mask = np.expand_dims(train_mask, axis=3)

    train_data = np.resize(train_data, (train_data.shape[0],512,512,1))
    train_mask = np.resize(train_mask, (train_mask.shape[0],512,512,1))

    test_data = np.expand_dims(test_data, axis=3)
    test_mask = np.expand_dims(test_mask, axis=3)

    test_data = np.resize(test_data, (test_data.shape[0],512,512,1))
    test_mask = np.resize(test_mask, (test_mask.shape[0],512,512,1))

    return train_data,train_mask,test_data,test_mask

#%%

model = unet()

model.summary()
model_checkpoint = ModelCheckpoint('unet_kidneys.hdf5', monitor='loss',verbose=1, save_best_only=True)
#%%
for i in range(15):
    train_data, train_mask, test_data, test_mask = prep_data(i)
    model.fit(x = train_data, y = train_mask,epochs=1,callbacks=[model_checkpoint], batch_size=2)

#%%
train_data,train_mask,test_data,test_mask = prep_data(11)
results = model.predict(x = test_data, batch_size=2 ,verbose=1)
#%%
slice_nr = 30
display([test_data[slice_nr,:,:,:], test_mask[slice_nr,:,:,:], results[slice_nr,:,:,:]]) 
 # %%
