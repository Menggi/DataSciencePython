import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import RandomForestClassifier

#Data load
ear1 = pd.read_csv('data-ear-1.csv')
ear2 = pd.read_csv('data-ear-2.csv')
ear3 = pd.read_csv('data-ear-3.csv')

landscape1= pd.read_csv('data-hand-landscape-1.csv')
landscape1.drop(landscape1[(landscape1.Type == 'MAGNETOMETER')
                            |(landscape1.Type == 'BAROMETER')].index, inplace=True)
landscape2= pd.read_csv('data-hand-landscape-2.csv')
landscape3= pd.read_csv('data-hand-landscape-3.csv')

portrait1 = pd.read_csv('data-hand-portrait-1.csv')
portrait2 = pd.read_csv('data-hand-portrait-2.csv')
portrait3 = pd.read_csv('data-hand-portrait-3.csv')

swing1 = pd.read_csv('data-hand-swinging-1.csv')
swing2 = pd.read_csv('data-hand-swinging-2.csv')
swing3 = pd.read_csv('data-hand-swinging-3.csv')

pocket1 = pd.read_csv('data-pocket-1.csv')
pocket2 = pd.read_csv('data-pocket-2.csv')
pocket3 = pd.read_csv('data-pocket-3.csv')

test1 = pd.read_csv('test-dataset-1.csv')
test2 = pd.read_csv('test-dataset-2.csv')

df_list = [ear1, ear2, ear3, landscape1, landscape2, landscape3, portrait1, portrait2,
      portrait3, swing1,swing2,swing3, pocket1, pocket2, pocket3]


#Feature function

def stft(x, size=256, overlap=2):   
    hop = size // overlap
    w = scipy.hanning(size+1)[:-1] 
    return np.array([np.fft.rfft(w*x.iloc[i:i+size]) for i in range(0, len(x)-size, hop)])


def feature_extract(df, size=256, overlap=2):
    meanamp = []
    stdamp = []
    rmsamp = []
    difamp = []    
    timeamp = []
    hop = size // overlap
    
    x = df['mean']
    stft_signal = stft(x)
    energy_signal = []
    
    for i, amp in enumerate(stft_signal):
        energy_signal.append(np.sum(np.power(abs(stft_signal[i]),2)))
    
    for i in range(0, len(x)-size, hop):
        meanamp.append(np.mean(x[i:i+size]))
        stdamp.append(np.std(x[i:i+size]))
        rmsamp.append(np.sqrt(np.sum(np.power(x[i:i+size],2))/size))
        difamp.append(np.max(x[i:i+size])-np.min(x[i:i+size]))  
        timeamp.append(np.max(df['Time (ms)'][i:i+size]))
                  
    return meanamp, stdamp, rmsamp, difamp, energy_signal, timeamp

i = 0

features = pd.DataFrame()
for df in df_list:
    #preprocessing
    temp = df[' X']*df[' X']+df[' Y']*df[' Y']+df[' Z']*df[' Z']
    df['mean'] = np.sqrt(temp)
    df_acc = df[df.Type == 'ACCELEROMETER'][['mean', 'Time (ms)']]
    df_gyro = df[df.Type == 'GYROSCOPE'][['mean', 'Time (ms)']]
    
    #smoothing    
    df_acc = df_acc.rolling(window=5, min_periods=1).mean()
    df_gyro =  df_gyro.rolling(window=5, min_periods=1).mean()
    
    #feature extraction
    mean_acc, std_acc, rms_acc, dif_acc, energy_acc, timeamp_acc= feature_extract(df_acc)
    mean_gyro, std_gyro, rms_gyro, dif_gyro, energy_gyro, timeamp_gyro= feature_extract(df_gyro)
    
    feature = pd.DataFrame(np.column_stack([mean_acc, std_acc, rms_acc, 
                                        dif_acc, energy_acc, mean_gyro, 
                                        std_gyro, rms_gyro, dif_gyro, energy_gyro]), 
                                   columns=list(range(1,11)))
    
    if i < 3:
        feature['label'] = 'ear'
    elif i < 6:
        feature['label'] = 'landscape'
    elif i < 9:
        feature['label'] = 'portrait'
    elif i < 12:
        feature['label'] = 'swing'
    else:
        feature['label'] = 'pocket'
    
    features = features.append(feature, ignore_index= True)
    
    i+=1

#Train the classifier
X = features[list(range(1,11))]
Y = features['label']

clf=RandomForestClassifier(n_estimators=100)

clf.fit(X, Y)

#Feature extraction from test data

test_list = [test1, test2]
test_result = pd.DataFrame()
for df in test_list:
    #preprocessing
    temp = df[' X']*df[' X']+df[' Y']*df[' Y']+df[' Z']*df[' Z']
    df['mean'] = np.sqrt(temp)
    df_acc = df[df.Type == 'ACCELEROMETER'][['mean', 'Time (ms)']]
    df_gyro = df[df.Type == 'GYROSCOPE'][['mean', 'Time (ms)']]
    
    #smoothing    
    df_acc = df_acc.rolling(window=5, min_periods=1).mean()
    df_gyro =  df_gyro.rolling(window=5, min_periods=1).mean()
    
    #feature extraction
    mean_acc, std_acc, rms_acc, dif_acc, energy_acc, time_acc= feature_extract(df_acc)
    mean_gyro, std_gyro, rms_gyro, dif_gyro, energy_gyro, time_gyro= feature_extract(df_gyro)
    
    feature = pd.DataFrame(np.column_stack([mean_acc, std_acc, rms_acc, 
                                        dif_acc, energy_acc, mean_gyro, 
                                        std_gyro, rms_gyro, dif_gyro, energy_gyro]), 
                                   columns=list(range(1,11)))
    
    #predict
    predict = clf.predict(feature)
    test = pd.DataFrame(
            {'Time(ms)' : time_acc,
             'label' : predict})
    test_result = test_result.append(test)
    
#Save result to .csv file
test_result.to_csv(r'result.csv', index = True)
    


