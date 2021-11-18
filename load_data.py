import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
seed = 456

def load_mnist(batchSize):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.

    # train val split
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=False, random_state=456)


    train_datasets = tf.data.Dataset.from_tensor_slices((x_tr, y_tr)).batch(batchSize).shuffle(x_tr.shape[0])
    val_datasets = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batchSize).shuffle(x_val.shape[0])
    test_datasets = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batchSize).shuffle(x_test.shape[0])
    return train_datasets, val_datasets, test_datasets

def genPerturbations(opt):
    X = np.tile(opt.canon4pts[:,0],[opt.batchSize,1])
    Y = np.tile(opt.canon4pts[:,1],[opt.batchSize,1])
    dX = tf.random.normal([opt.batchSize,4], seed=seed)*opt.pertScale + tf.random.normal([opt.batchSize,1], seed=seed)*opt.transScale
    dY = tf.random.normal([opt.batchSize,4], seed=seed)*opt.pertScale + tf.random.normal([opt.batchSize,1], seed=seed)*opt.transScale
    O = np.zeros([opt.batchSize, 4], dtype=np.float32)
    I = np.ones([opt.batchSize, 4], dtype=np.float32)

    # fit warp parameters to generated displacements
    if opt.warpType=="homography":
        A = tf.concat([tf.stack([X,Y,I,O,O,O,-X*(X+dX),-Y*(X+dX)],axis=-1),
                       tf.stack([O,O,O,X,Y,I,-X*(Y+dY),-Y*(Y+dY)],axis=-1)],1)
        b = tf.expand_dims(tf.concat([X+dX,Y+dY],1),-1)
        pPert = tf.compat.v1.matrix_solve(A,b)[:,:,0]
        pPert -= tf.cast([[1,0,0,0,1,0,0,0]], tf.float32)
    else:
        if opt.warpType=="translation":
            J = np.concatenate([np.stack([I,O],axis=-1),
                                np.stack([O,I],axis=-1)],axis=1)
        if opt.warpType=="similarity":
            J = np.concatenate([np.stack([X,Y,I,O],axis=-1),
                                np.stack([-Y,X,O,I],axis=-1)],axis=1)
        if opt.warpType=="affine":
            J = np.concatenate([np.stack([X,Y,I,O,O,O],axis=-1),
                                np.stack([O,O,O,X,Y,I],axis=-1)],axis=1)
        dXY = tf.expand_dims(tf.concat([dX,dY],1),-1)
        pPert = tf.compat.v1.matrix_solve_ls(J,dXY)[:,:,0]
    return pPert