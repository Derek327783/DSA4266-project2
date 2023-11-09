import pandas as pd
import numpy as np
import argparse
import gzip
import json
from tensorflow.keras.models import load_model

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
    parser = argparse.ArgumentParser(description='predict with the neural networks')
    parser.add_argument('--model', type=str, default='./model/best.h5', help='model path')
    parser.add_argument('--data', type=str, default= './data/dataset0.json.gz', help='predict dataset path')
    parser.add_argument('--data_output', type=str, default='./output/output.csv', help='predict data have label or not')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    parsed_data_list = []

    # Read and decompress the file, parsing each line as a separate JSON object
    with open(args.data, 'r') as json_file:
        for line in json_file:
            try:
                data = json.loads(line)
                # Append the parsed JSON object to the list
                for i in data.keys():
                    for j in data[i].keys():
                        for z in data[i][j].keys():                  
                            parsed_data_list.append([i, j, z, data[i][j][z]])
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line: {line}")
    data = pd.DataFrame(parsed_data_list)
    #data = data.rename(columns={'0': 'Transcript_ID', '1': 'Position',"2":"Base_seq","3":"Sample_reads"})
    data.columns=['Transcript_ID', 'Position', 'Base_seq', 'Sample_reads']

    #print(data.head())
    # Mean and std
    def feature_engin1(rows):
        sample_reads = rows["Sample_reads"]
        output = {"Mean":[],"Standard_deviation":[]}
        matrix = np.array(sample_reads)
        column_means = np.mean(matrix, axis=0)
        column_std = np.std(matrix,axis=0)
        output = [list(column_means),list(column_std)]
        return output
    
    data["Compiled_reads"] = data.apply(feature_engin1,axis=1)

    # Splitting the compiled reads into separate columns
    new_columns = data['Compiled_reads'].apply(pd.Series).apply(pd.Series)
    new_columns = pd.concat([new_columns[0].apply(pd.Series), new_columns[1].apply(pd.Series)], axis = 1)
    new_columns.columns = ['mean1', 'mean2', 'mean3', 'mean4', 'mean5', 'mean6', 'mean7', 'mean8', 'mean9',
                'sd1', 'sd2', 'sd3', 'sd4', 'sd5', 'sd6', 'sd7', 'sd8', 'sd9']
    
    merged_right =  data

    merged_right_processed = pd.concat([merged_right,new_columns],axis = 1)
    merged_right_processed = merged_right_processed.drop(columns = ["Sample_reads","Compiled_reads"])
    #print(merged_right_processed.head())
    #test_data = new_columns.drop(columns = ['Unnamed: 0'])
    test_data = merged_right_processed

    one_hot_encode_test_OG = test_data.copy()

    one_hot_encode_test_OG["One_hot_encode_base_seq"] = one_hot_encode_test_OG.apply(base_seq_one_hot,axis=1)

    new_column_test = one_hot_encode_test_OG['One_hot_encode_base_seq'].apply(pd.Series)

    new_column_test.columns = ['1A', '1C', '1G', '1T','2A', '2C', '2G', '2T','3A', '3C', '3G', '3T'
                                ,'4A', '4C', '4G', '4T','5A', '5C', '5G', '5T','6A', '6C', '6G', '6T'
                                ,'7A', '7C', '7G', '7T']

    one_hot_encode_test_OG = pd.concat([one_hot_encode_test_OG,new_column_test],axis = 1)

    X_test = one_hot_encode_test_OG.drop(columns = ["One_hot_encode_base_seq", "Transcript_ID","Base_seq","Position"],axis = 1)

    model = load_model(args.model)

    Y_pred = model.predict(X_test)
    df = pd.DataFrame()
    df['Transcript_id'] = data['Transcript_ID']
    df['Position'] = data['Position']
    df['Score'] = Y_pred
    df.to_csv(args.data_output, index=False)
