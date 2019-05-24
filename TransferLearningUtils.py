import numpy as np

import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Activation, Lambda
from keras import backend as K

#can be needed for older versions of tensrflow
def MyAddAxes(x):
    return K.expand_dims(K.expand_dims(x,axis=-1),axis=-1)
    
#my_preprocess_numpy_input replaces the calls to preprocess input from the keras_applications.imagenet_utils library
def MyPreprocessInput(x, data_format, mode, **kwargs):
    """Preprocesses a Numpy array encoding a batch of images.
    # Arguments
        x: Input array, 3D or 4D.
        data_format: Data format of the image array.
        mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                will normalize each channel with respect to the
                ImageNet dataset.
    # Returns
        Preprocessed Numpy array.
    """
    #backend, _, _, _ = get_submodules_from_kwargs(kwargs)
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(backend.floatx(), copy=False)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
        return x

    if mode == 'torch':
        x /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        #caffe
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            if x.ndim == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        std = None

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        for i in range(x.shape[-1]):
            x[...,i] -= np.mean(mean)
            x[...,i] /= np.mean(std)
        #x[..., 0] -= mean[0]
        #x[..., 1] -= mean[1]
        #x[..., 2] -= mean[2]
        #if std is not None:
        #    x[..., 0] /= std[0]
        #    x[..., 1] /= std[1]
        #    x[..., 2] /= std[2]
    return x
    
def ChangeModelShape(ModelName = None,
                     NumInputChannels=3,
                     Pretrained=True,
                     FirstLayerInit = 0,
                     NumClasses=2,
                     wd=1e-5):
    
    #FirstLayerInit == 0 means randomly initialise the first layer weights
    #FirstLayerInit == 1 means use repeated average of pretrained first layer weights
    #FirstLayerInit is ignored if Pretrained==False
    
    #there are two parts to this code: 
    #first, change the number of input channels if not 3
    #   and copy the pretrained weights into the new model, if Pretrained==True
    #second, add the output layers to change the number of classes
    
    #part 1: input channels
    if Pretrained==True:
        base_pretrained_model = ModelName(weights='imagenet', include_top=False)
        if NumInputChannels == 3:
            output_model = base_pretrained_model
        else:
            output_model = ModelName(weights=None,
                                      include_top=False,
                                      input_shape=(None,None,NumInputChannels))

            #get the weights out of the pretrained model (3 input channels)  
            pretrained_weights = base_pretrained_model.get_weights()

            #set the first layer weights
            new_weights = []
            
            if FirstLayerInit == 1:
                #average the 3 pretrained channels and use in the first layer of new model
                new_layer1_weights = np.mean(pretrained_weights[0], axis=2)[:, :, None, :]
                if NumInputChannels > 1:
                    new_layer1_weights = np.repeat(new_layer1_weights, 
                                                   NumInputChannels, 
                                                   axis=2)
                new_weights.append(new_layer1_weights)
            elif FirstLayerInit == 0:
                #use the random weights provided in the randomly initialized model
                new_weights.append(output_model.get_weights()[0])
            
            #copy the pretrained weights into the new model, for all except the first layer
            for i in range(1, len(pretrained_weights)):
                new_weights.append(pretrained_weights[i])

            #copy all the pretrained weights into the new  model
            output_model.set_weights(new_weights)
    else:
        output_model = ModelName(weights=None,
                                      include_top=False,
                                      input_shape=(None,None,NumInputChannels))
    
    #part 2: add an output for specified number of classes
    # add a global spatial average pooling layer
    x = output_model.output
    x = GlobalAveragePooling2D()(x)
    if tf.__version__=="1.4.0":
        x = Lambda(MyAddAxes)(x)
    predictions = Dense(NumClasses,
                        activation='softmax',
                        kernel_initializer = 'he_normal',
                        kernel_regularizer=l2(wd),
                        use_bias = False
                       )(x)
    return Model(inputs=output_model.input, outputs=predictions)
