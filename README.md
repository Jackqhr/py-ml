# Tensorflow

> load model

```python
import tensorflow as tf
path = '/kaggle/input/mnisttf/mnist.npz'
(x_train,y_train),(x_show,y_test)=mnist.load_data(path=path)
x_train=x_train.reshape(60000,784).astype('float32')/255.0
x_test=x_show.reshape(10000,784).astype('float32')/255.0
model=tf.keras.models.load_model(r'data_model/xxxx.h5')
k=1
pre=x_test[k].reshape(1,784)
print(model.predict(pre))
```