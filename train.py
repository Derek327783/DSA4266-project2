import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from imblearn.over_sampling import ADASYN
import argparse

#Balancing the data
def random_undersample(train_data):
    rus = RandomUnderSampler()
    X = train_data.drop(columns = ["label","gene_id","Transcript_ID"])
    Y = train_data['label'] 
    X_resampled, y_resampled = rus.fit_resample(X, Y)
    output = pd.concat([X_resampled,y_resampled],axis = 1) 
    return output


def GANsampling(train_data,no_of_samples = 50000):
    data = train_data.drop(columns = ["gene_id","Transcript_ID"])
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=data)
    metadata.update_column("label",sdtype="categorical")
    CTGAN = CTGANSynthesizer(
        metadata, 
        enforce_rounding=False,
        epochs=50,
        verbose=False
    )
    CTGAN.fit(data)
    CT_samples = CTGAN.sample(num_rows=no_of_samples)
    New_positive = CT_samples[CT_samples["label"] == 1]
    output = pd.concat([data,New_positive],axis = 0, ignore_index = True) 
    return output

def adasyn_sample(train):
    X = train.drop(columns = ["label","gene_id","Transcript_ID","Base_seq"])
    y = train['label']
    adasyn_model = ADASYN(sampling_strategy='auto', random_state=42)
    X_adasyn_sample, y_adasyn_sample = adasyn_model.fit_resample(X, y)
    output = pd.concat([X_adasyn_sample, y_adasyn_sample], axis=1)
    return output

#One hot for base sequence
def base_seq_one_hot(rows):
    base_seq = rows["Base_seq"]
    output = [0] * 28
    for i in range(0,7):
        base = base_seq[i]
        if base == "A":
            output[(i+1)*4 - 4] = 1
        elif base == "C":
            output[(i+1)*4 - 3] = 1
        elif base == "G":
            output[(i+1)*4 - 2] = 1
        else:
            output[(i+1)*4 - 1] = 1
    return output

#get arguments
def get_args():
    parser = argparse.ArgumentParser(description='Train the neural networks')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--data', type=str, default= './data/train_OG.csv', help='train dataset path')
    parser.add_argument('--test_data', type=str, default= './data/test_OG.csv', help='test dataset path')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    #Preprocess the dataset
    train_data =  pd.read_csv(args.data)
    test_data =  pd.read_csv(args.test_data)
    train_data = train_data.drop(columns = ['Unnamed: 0'])
    test_data = test_data.drop(columns = ['Unnamed: 0'])

    one_hot_encode_train_OG = random_undersample(train_data)
    one_hot_encode_test_OG = test_data.copy()

    one_hot_encode_train_OG["One_hot_encode_base_seq"] = one_hot_encode_train_OG.apply(base_seq_one_hot,axis=1)
    one_hot_encode_test_OG["One_hot_encode_base_seq"] = one_hot_encode_test_OG.apply(base_seq_one_hot,axis=1)

    new_column_train = one_hot_encode_train_OG['One_hot_encode_base_seq'].apply(pd.Series)
    new_column_test = one_hot_encode_test_OG['One_hot_encode_base_seq'].apply(pd.Series)
    new_column_train.columns = ['1A', '1C', '1G', '1T','2A', '2C', '2G', '2T','3A', '3C', '3G', '3T'
                                ,'4A', '4C', '4G', '4T','5A', '5C', '5G', '5T','6A', '6C', '6G', '6T'
                                ,'7A', '7C', '7G', '7T']

    new_column_test.columns = ['1A', '1C', '1G', '1T','2A', '2C', '2G', '2T','3A', '3C', '3G', '3T'
                                ,'4A', '4C', '4G', '4T','5A', '5C', '5G', '5T','6A', '6C', '6G', '6T'
                                ,'7A', '7C', '7G', '7T']

    one_hot_encode_train_OG = pd.concat([one_hot_encode_train_OG,new_column_train],axis = 1)
    one_hot_encode_test_OG = pd.concat([one_hot_encode_test_OG,new_column_test],axis = 1)

    X_train = one_hot_encode_train_OG.drop(['label',"Position","Base_seq","One_hot_encode_base_seq"], axis=1)
    Y_train = one_hot_encode_train_OG['label']
    X_test = one_hot_encode_test_OG.drop(columns = ["One_hot_encode_base_seq","label","gene_id","Transcript_ID","Base_seq","Position"],axis = 1)
    Y_test = one_hot_encode_test_OG['label']


    #Training
    num = 2
    def lr_schedule(epoch):
        if epoch < 25 * num:
            return 0.01
        elif epoch < 50 * num:
            return 0.001
        elif epoch < 75 * num:
            return 0.0001
        else:
            return 0.00001
        
    # Create a custom learning rate callback
    learning_rate_scheduler = LearningRateScheduler(lr_schedule)

    # Create a sequential neural network model
    model = keras.Sequential()

    # Add input layer

    #GAN with one hot
    model.add(keras.layers.Input(shape=(46,)))  # 18 input features

    #GAN
    #model.add(keras.layers.Input(shape=(19,)))
    # Add hidden layers
    '''
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.BatchNormalization())

    #model.add(keras.layers.Dense(128, activation=keras.layers.LeakyReLU(alpha=0.2)))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    #model.add(keras.layers.Dropout(0.5))
    '''

    #model.add(keras.layers.Dense(64, activation='relu'))
    #model.add(keras.layers.Dense(64, activation=keras.layers.LeakyReLU(alpha=0.2)))
    #model.add(keras.layers.BatchNormalization())
    #model.add(keras.layers.Dropout(0.5))

    #model.add(keras.layers.Dense(16, activation='relu'))
    #model.add(keras.layers.BatchNormalization())

    #model.add(keras.layers.Dense(32, activation='relu'))
    #model.add(keras.layers.BatchNormalization())

    # model.add(keras.layers.Dense(64, activation='relu'))
    # model.add(keras.layers.BatchNormalization())

    #model.add(keras.layers.Dense(32, activation=keras.layers.LeakyReLU(alpha=0.2)))
    #model.add(keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    #model.add(keras.layers.Dropout(0.5))

    #model.add(keras.layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    #model.add(keras.layers.Dropout(0.5))
    # Add output layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))  # Binary classification, so use 'sigmoid' activation

    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, Y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(X_test, Y_test), callbacks=[learning_rate_scheduler])
    #model.fit(X_train, Y_train, epochs=20, batch_size=32)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

    # Make predictions
    Y_pred = model.predict(X_test)

    #Save the model
    model.save('./model/nn_gene.h5')

    #Create result plot
    fpr, tpr, _ = roc_curve(Y_test, Y_pred)
    roc_auc = auc(fpr, tpr)
    #accuracy = accuracy_score(Y_test, Y_pred)
    #print(f'Accuracy: {accuracy * 100:.2f}%')
    print(roc_auc)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    plt.savefig('plot.png')

    precision, recall, thresholds = precision_recall_curve(Y_test, Y_pred)

    # Calculate the average precision score
    average_precision = average_precision_score(Y_test, Y_pred)
    print(average_precision)

    # Plot the Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve (AP = {average_precision:.2f})')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig("pr.png")
