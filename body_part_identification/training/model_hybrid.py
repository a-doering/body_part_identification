from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import(
        Conv2D,
        MaxPooling2D,
        Reshape
        )
from tensorflow.python.keras.layers import (
    Input,
    Dense,
    Flatten
)

from tensorflow.python.keras import backend as K
K.set_image_data_format = 'channels_last'



def build_model(patch_size, n_classes):
    '''
    Model is build after the single stage model of Zhang
    Changed output shape and removed one dense layer at the end    
    '''
    input = Input(shape=patch_size, name='input_layer')
    n_base_filter = 32
    reshaped = Reshape([patch_size[1],patch_size[2],1])(input)

    # Some convolutional layers
    conv_1 = Conv2D(n_base_filter,
                                    kernel_size=(2, 2),
                                    padding='same',
                                    activation='relu')(reshaped)
    conv_2 = Conv2D(n_base_filter,
                                    kernel_size=(2, 2),
                                    padding='same',
                                    activation='relu')(conv_1)
    conv_2 = MaxPooling2D(pool_size=(3, 3), padding='same')(conv_2)
    
    # Some convolutional layers
    conv_3 = Conv2D(n_base_filter*2,
                                    kernel_size=(2, 2),
                                    padding='same',
                                    activation='relu')(conv_2)
    conv_4 = Conv2D(n_base_filter*2,
                                    kernel_size=(2, 2),
                                    padding='same',
                                    activation='relu')(conv_3)
    conv_4 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv_4)
    
    # Now layers 8-12 in Philips net, no pooling at the end
    conv_5 = Conv2D(n_base_filter*4,
                                    kernel_size=(2, 2),
                                    padding='same',
                                    activation='relu')(conv_4)
    conv_6 = Conv2D(n_base_filter*8,
                                    kernel_size=(2, 2),
                                    padding='same',
                                    activation='relu')(conv_5)
    conv_6 = MaxPooling2D(pool_size=(2, 2),
                                          padding='same')(conv_6)
    
    
    conv_7 = Conv2D(n_base_filter*16,
                                    kernel_size=(2, 2),
                                    padding='same',
                                    activation='relu')(conv_6)
    
    conv_8 = Conv2D(n_base_filter*32,
                                    kernel_size=(2, 2),
                                    padding='same',
                                    activation='relu')(conv_7)
    
    dense_2 = Dense(4096, activation='relu')(conv_8)
    
    flattening_1 = Flatten()(dense_2)
    dense_3 = Dense(n_classes, activation='relu')(flattening_1)
    landmark_class_probability = Dense(n_classes, activation='softmax', name='class')(dense_3)
    direct_regression = Dense(1, activation='linear', name='reg')(dense_3)
    # Wrap in a Model
    model = Model(inputs=input, outputs=[landmark_class_probability, direct_regression])
    return model


if __name__ == '__main__':
    n_classes = 5
    patch_size =[1,40, 16]
    input =  [40,16,1]
    from keras.utils import plot_model
    plot_model(build_model(patch_size, n_classes), to_file='model.png')
    my_model = build_model(patch_size, n_classes)
    print(my_model.summary())
