import os
for filename in range(100):
    filename = str(filename)
    if os.path.isfile('img/out/'+filename+'/'+filename+'_predict_result.png'):
        os.remove('img/out/'+filename+'/'+filename+'_predict_result.png')