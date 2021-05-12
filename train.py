import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
import random
from watermark import watermarking, gaussian_noise
from evaluate import *


SEED=1011


def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def training(args, train_x, train_x_wm, train_y, test_x, test_x_wm, test_y):
    
    set_seeds()
    
    BATCH_SIZE=2048
    
    saved_model_path="./load_model/classifier/"+str(args.dataset)+"/"+str(args.model)+"_SP_"+str(args.alpha)
    forgery_directory="./data/forgery/"+str(args.dataset)
    
    if args.dataset=='cifar10':
        n_labels=10
    elif args.dataset=='cifar100':
        n_labels=20
        
    if not os.path.exists(saved_model_path+'/saved_model.pb'):

        input_tensor = tf.keras.Input(shape=(32, 32, 3))
        resized_images = layers.Lambda(lambda image: tf.image.resize(image, (224, 224)))(input_tensor)

        if args.model=='resnet':
            base_model = tf.keras.applications.ResNet152V2(
                include_top=False,
                weights='imagenet',
                input_tensor=resized_images,
                input_shape=(224, 224, 3),
            )

        elif args.model=='densenet':
            base_model = tf.keras.applications.DenseNet201(
                include_top=False,
                weights='imagenet',
                input_tensor=resized_images,
                input_shape=(224, 224, 3),
            )

        for layer in base_model.layers:
            layer.trainable = False

        outputs = base_model.layers[-1].output
        
        output = layers.GlobalAveragePooling2D()(outputs)
        output = layers.Dense(1024, activation='gelu', kernel_initializer="he_normal")(output)
        output = layers.Dropout(0.2)(output)
        output = layers.Dense(256, activation='gelu', kernel_initializer="he_normal")(output)
        output = layers.Dropout(0.2)(output)
        output = layers.Dense(64, activation='gelu', kernel_initializer="he_normal")(output)
        output = layers.Dropout(0.2)(output)
        output = layers.Dense(n_labels, activation='softmax', name='logits_out')(output)
   
        output2 = layers.Conv2DTranspose(512, 3, strides=2, padding='same',
                                         kernel_initializer="he_normal")(outputs)
        output2 = layers.BatchNormalization()(output2)
        output2 = layers.Activation('gelu')(output2)
        output2 = layers.Conv2DTranspose(128, 3, strides=2, padding='same',
                                         kernel_initializer="he_normal")(output2)
        output2 = layers.BatchNormalization()(output2)
        output2 = layers.Activation('gelu')(output2)
        output2 = layers.Conv2DTranspose(32, 3, strides=1,
                                         kernel_initializer="he_normal")(output2)
        output2 = layers.BatchNormalization()(output2)
        output2 = layers.Activation('gelu')(output2)
        output2 = layers.Conv2DTranspose(3, 3, strides=1,
                                         activation='sigmoid', name='rec_output')(output2)


        base_model = tf.keras.models.Model(inputs=input_tensor, outputs=[output, output2])
        
        base_model.compile(optimizer='adam',
                           loss={'logits_out' : 'sparse_categorical_crossentropy',
                                 'rec_output' : 'mean_squared_error'},
                           loss_weights={'logits_out': 0.9,
                                         'rec_output': 0.1})
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                    restore_best_weights=True)
        
        base_model.fit(train_x_wm,
                       [train_y, train_x],
                       validation_split=0.2,
                       epochs=50,
                       batch_size=BATCH_SIZE,
                       callbacks=[callback])
        
        base_model.save(saved_model_path)
        base_model = tf.keras.models.load_model(saved_model_path)
        
    else:
        base_model = tf.keras.models.load_model(saved_model_path)
        
        
    if not os.path.exists(saved_model_path+"_OE"+'/saved_model.pb'):
        
        aux_model = tf.keras.models.load_model(saved_model_path)
        
        if args.model == 'resnet':
            for layer in aux_model.layers[0:565]:
                layer.trainable = False
        elif args.model == 'densenet':
            for layer in aux_model.layers[0:708]:
                layer.trainable = False
        
        prob_in, pseudo_image = aux_model.predict(train_x_wm)
        
        pseudo_out=watermarking(pseudo_image, args.alpha)
        
        ## Gaussian noise
#         pseudo_out=random_noise(pseudo_image, mode="gaussian", var=args.alpha)
        ## Salt-and-pepper noise
#         pseudo_out=random_noise(pseudo_image, mode="s&p", amount=args.alpha)
        
        uniform=np.expand_dims(np.repeat(1/n_labels, n_labels), axis=1)
        pseudo_prob_out=np.repeat(uniform, len(pseudo_out), axis=1).T
        
        train_x_aux=np.concatenate([train_x, pseudo_out], axis=0)
        train_y_aux=np.concatenate([prob_in, pseudo_prob_out], axis=0)
        
        aux_model.compile(optimizer='adam',
                          loss={'logits_out' : 'kl_divergence',
                                 'rec_output' : 'mean_squared_error'},
                          loss_weights={'logits_out': 0.9,
                                         'rec_output': 0.1})
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                    restore_best_weights=True)
       
        aux_model.fit(train_x_aux,
                      [train_y_aux, train_x_aux],
                      validation_split=0.2,
                      epochs=100,
                      batch_size=BATCH_SIZE,
                      callbacks=[callback])
        
        aux_model.save(saved_model_path+"_OE")
        aux_model = tf.keras.models.load_model(saved_model_path+"_OE")
        
    else:
        aux_model = tf.keras.models.load_model(saved_model_path+"_OE")
        
        
    print("=== Evaluate ===")

    forgery = tf.keras.preprocessing.image_dataset_from_directory(forgery_directory,
                                                                  batch_size=10000,
                                                                  image_size=(32, 32))
    forgery=list(forgery.as_numpy_iterator())[0]
    forgery_x=forgery[0].astype("uint8")
    forgery_x=forgery_x / 255.
    forgery_y=forgery[1]
    
    test_y=test_y.reshape(-1)
    
    s_prob_right, s_prob_wrong, kl_right, kl_wrong =\
        right_wrong_distinction(base_model, test_x, test_y)
    
    s_prob_right, s_prob_wrong, kl_right, kl_wrong =\
        right_wrong_distinction(aux_model, test_x, test_y)
    
    s_prob_in_f, s_prob_out_f, pseudo_prob_in_f, pseudo_prob_out_f =\
        in_out_distinction(base_model, aux_model, test_x, forgery_x)
    
    
    print("\n=== The End ===")