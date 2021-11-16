import tensorflow as tf
import pickle

present_model = tf.keras.models.load_model('Weight/diabetes.h5')

# 7,97,76,32,91,40.9,0.871,32,1  == 0.86088324 V
# 2,68,62,13,15,20.1,0.257,23,0  == 0.0 V
# 3,103,72,30,152,27.6,0.73,27,0 == 0.024339795 V
# 11,136,84,35,130,28.3,0.26,42,1 == 0.3616997 X
# 1,91,54,25,100,25.2,0.234,23,0 == 1.9289608e-32 V
# 2,155,52,27,540,38.7,0.24,25,1 == 0.86088324 V
# 1,149,68,29,127,29.3,0.349,42,1 == 0.7537323 V
# 3,173,78,39,185,33.8,0.97,31,1 == 0.86088324 V

# 11,120,80,37,150,42.3,0.785,48,1 == 0.86088324 V
# 3,102,44,20,94,30.8,0.4,26,0 == 7.2178665e-21 V
# 1,109,58,18,116,28.5,0.219,22,0 == 1.2308042e-31 V
# 13,153,88,37,140,40.6,1.174,39,0 == 0.86088324 X
# 12,100,84,33,105,30,0.488,46,0 == 1.7319336e-07 V
# 1,81,74,41,57,46.3,1.096,32,0 == 0.0025134087 V
# 3,187,70,22,200,36.4,0.408,36,1 == 0.86088324 V
# 1,121,78,39,74,39,0.261,28,0 == 0.000770241 V
# 0,181,88,44,510,43.3,0.222,26,1 == 0.86088324 V
# 1,128,88,39,110,36.5,1.057,37,1 == 0.86088324 V

std = pickle.load(open('std.pkl', 'rb'))
# model = pickle.load(open("svm_model.pkl", "rb"))

print(std)
# row_df0 = std.transform([[11,120,80,37,150,42.3,0.785,48]])
# row_df1 = std.transform([[3,102,44,20,94,30.8,0.4,26]])
# row_df2 = std.transform([[1,109,58,18,116,28.5,0.219,22]])
# row_df3 = std.transform([[13,153,88,37,140,40.6,1.174,39]])
# row_df4 = std.transform([[12,100,84,33,105,30,0.488,46]])
# row_df5 = std.transform([[1,81,74,41,57,46.3,1.096,32]])
# row_df6 = std.transform([[3,187,70,22,200,36.4,0.408,36]])
# row_df7 = std.transform([[1,121,78,39,74,39,0.261,28]])
# row_df8 = std.transform([[0,181,88,44,510,43.3,0.222,26]])
# row_df9 = std.transform([[1,128,88,39,110,36.5,1.057,37]])


# arr = [row_df0, row_df1, row_df2, row_df3, row_df4,
#        row_df5, row_df6, row_df7, row_df8, row_df9]

# for i in range(len(arr)):
    
#     ouput = present_model.predict(arr[i])
#     print(ouput[0][0])


# row_df0 = std.transform([[7, 97, 76, 32, 91, 40.9, 0.871, 32]])
# row_df1 = std.transform([[2, 68, 62, 13, 15, 20.1, 0.257, 23]])
# row_df2 = std.transform([[3, 103, 72, 30, 152, 27.6, 0.73, 27]])
# row_df3 = std.transform([[11, 136, 84, 35, 130, 28.3, 0.26, 42]])
# row_df4 = std.transform([[1, 91, 54, 25, 100, 25.2, 0.234, 23]])
# row_df5 = std.transform([[2,155,52,27,540,38.7,0.24,25]])
# row_df6 = std.transform([[1,149,68,29,127,29.3,0.349,42]])
# row_df7 = std.transform([[3, 173, 78, 39, 185, 33.8, 0.97, 31]])


# arr = [row_df0, row_df1, row_df2, row_df3, row_df4,
#        row_df5, row_df6, row_df7]

# for i in range(len(arr)):

#     prediction = model.predict_proba(arr[i])
#     output = '{0:.{1}f}'.format(prediction[0][1], 2)
#     output_print = str(float(output)*100)+'%'
#     print(output_print)
