import time, datetime
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import tensorflow as tf

import options
import network
import load_data
import warp
import util

print(util.toYellow("======================================================="))
print(util.toYellow("main.py (training on MNIST)"))
print(util.toYellow("======================================================="))

class Project():
    def __init__(self):

        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print(self.current_time)
        self.opt = options.set(training=True)

        self.train_datasets, self.val_datasets, self.test_datasets = load_data.load_mnist(batchSize=self.opt.batchSize)

        # -------------------- Create model --------------------
        self.IC_STN = network.IC_STN(self.opt.warpDim)
        self.CNN = network.CNN(self.opt.labelN)

        self.op_IC_STN = tf.optimizers.SGD(self.opt.lrGP, momentum=0.8)
        self.op_CNN = tf.optimizers.SGD(self.opt.lrC, momentum=0.8)

        # -------------------- Create writer --------------------
        self.checkpoint_dir = f'./checkpoints/{self.current_time}'

        self.train_writer = tf.summary.create_file_writer(f'./logs/{self.current_time}/train')
        self.valid_writer = tf.summary.create_file_writer(f'./logs/{self.current_time}/valid')

        self.checkpoint = tf.train.Checkpoint(optimizer_GP=self.op_IC_STN,
                                              optimizer_C=self.op_CNN,
                                              model_GP=self.IC_STN,
                                              model_C=self.CNN)

        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_dir, max_to_keep=30)

        # plot example
        for test_x, _ in self.test_datasets:
            self.testing_example = test_x[0:5]
            break

    def calculate_loss_acc(self, y_true, y_pred):
        labelOnehot = tf.one_hot(y_true, self.opt.labelN)
        loss = tf.keras.losses.categorical_crossentropy(y_pred, labelOnehot)
        loss = tf.reduce_mean(loss)

        y_pred = tf.argmax(y_pred, axis=-1)
        acc = accuracy_score(y_true, y_pred)
        return loss, acc

    def train_step(self, p, images, label):
        with tf.GradientTape() as GP_tape, tf.GradientTape() as C_tape:
            imageWarpAll = []
            for l in range(self.opt.warpN):
                pMtrx = warp.vec2mtrx(self.opt, p)
                imageWarp, _, _ = warp.transformImage(self.opt, images, pMtrx)
                imageWarpAll.append(imageWarp)
                feat = self.IC_STN(imageWarp)
                dp = feat
                p = warp.compose(self.opt, p, dp)
            pMtrx = warp.vec2mtrx(self.opt, p)
            imageWarp, _, _ = warp.transformImage(self.opt, images, pMtrx)
            imageWarpAll.append(imageWarp)
            imageWarp = imageWarpAll[-1]
            predictions = self.CNN(imageWarp)
            loss, acc = self.calculate_loss_acc(label, predictions)
        gradients_of_GP = GP_tape.gradient(loss, self.IC_STN.trainable_variables)
        gradients_of_C = C_tape.gradient(loss, self.CNN.trainable_variables)
        self.op_IC_STN.apply_gradients(zip(gradients_of_GP, self.IC_STN.trainable_variables))
        self.op_CNN.apply_gradients(zip(gradients_of_C, self.CNN.trainable_variables))
        return loss, acc

    def test_step(self, p, images, label=None):
        imageWarpAll = []
        for l in range(self.opt.warpN):
            pMtrx = warp.vec2mtrx(self.opt, p)
            imageWarp, _, _ = warp.transformImage(self.opt, images, pMtrx)
            imageWarpAll.append(imageWarp)
            feat = self.IC_STN(imageWarp)
            dp = feat
            p = warp.compose(self.opt, p, dp)
        pMtrx = warp.vec2mtrx(self.opt, p)
        imageWarp, XfloorInt, YfloorInt = warp.transformImage(self.opt, images, pMtrx)
        imageWarpAll.append(imageWarp)
        imageWarp = imageWarpAll[-1]
        predictions = self.CNN(imageWarp)
        loss, acc = 0, 0
        if label != None:
            loss, acc = self.calculate_loss_acc(label, predictions)
        return loss, acc, imageWarp, XfloorInt, YfloorInt

    def plot_Ori_region(self, Ori_image, imageWarp, X_coordinate, Y_coordinate, epoch, pad_width=8):

        epoch = epoch + 1
        print('plot')
        util.mkdir(f'./testing_plot')
        util.mkdir(f'./testing_plot/{self.current_time}')
        for i in range(Ori_image.shape[0]):
            util.mkdir(f'./testing_plot/{self.current_time}/test{i + 1}')

            X_coord = X_coordinate[i + 1] + pad_width
            Y_coord = Y_coordinate[i + 1] + pad_width

            X_coord = tf.reshape(X_coord, [784])
            Y_coord = tf.reshape(Y_coord, [784])

            x1, x2, x3, x4 = X_coord[0], X_coord[27], X_coord[756], X_coord[783]
            y1, y2, y3, y4 = Y_coord[0], Y_coord[27], Y_coord[756], Y_coord[783]

            ori = Ori_image[i].numpy().reshape((28, 28)) * 255.
            padding = np.pad(ori, pad_width, 'constant', constant_values=255)
            img = tf.reshape(tf.cast(padding, tf.uint8), [28 + pad_width * 2, 28 + pad_width * 2, 1])

            plt.figure()
            plt.imshow(img, cmap="gray")
            plt.plot([x1, x2], [y1, y2], color='red')
            plt.plot([x2, x4], [y2, y4], color='red')
            plt.plot([x3, x4], [y3, y4], color='red')
            plt.plot([x3, x1], [y3, y1], color='red')
            plt.savefig(f'./testing_plot/{self.current_time}/test{i + 1}/epoch{epoch}_ori')
            plt.close('all')

            plt.figure()
            img = tf.cast(tf.reshape(imageWarp[i] * 255., [28, 28]), tf.uint8)
            plt.imshow(img, cmap='gray')
            plt.savefig(f'./testing_plot/{self.current_time}/test{i + 1}/epoch{epoch}_warp')
            plt.close('all')

    def run(self):
        for epoch in range(self.opt.epochs):
            start = time.time()
            # ---------training---------
            L_tr = []
            A_tr = []
            for image_batch, label_batch in self.train_datasets:
                Init_p = load_data.genPerturbations(self.opt)
                loss_tr, acc_tr = self.train_step(Init_p, image_batch, label_batch)
                L_tr.append(loss_tr)
                A_tr.append(acc_tr)

            # ---------validation---------
            L_val = []
            A_val = []
            for image_batch, label_batch in self.val_datasets:
                Init_p = load_data.genPerturbations(self.opt)
                loss_val, acc_val, _, _, _ = self.test_step(Init_p, image_batch, label_batch)
                L_val.append(loss_val)
                A_val.append(acc_val)

            with self.train_writer.as_default():
                tf.summary.scalar('loss(epoch)',  tf.reduce_mean(L_tr), step=epoch + 1)
                tf.summary.scalar('acc(epoch)',  tf.reduce_mean(A_tr), step=epoch + 1)

            with self.valid_writer.as_default():
                tf.summary.scalar('loss(epoch)', tf.reduce_mean(L_val), step=epoch + 1)
                tf.summary.scalar('acc(epoch)', tf.reduce_mean(A_val), step=epoch + 1)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
            template = 'Train Loss: {}, Train Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
            print(template.format(tf.reduce_mean(L_tr).numpy(), tf.reduce_mean(A_tr).numpy(),
                                  tf.reduce_mean(L_val).numpy(), tf.reduce_mean(A_val).numpy()))


            # # update learning rate
            # # lrGP = self.opt.lrGP * (self.opt.lrGPdecay ** tf.cast((iter // self.opt.lrGPstep), tf.float32))
            # # lrC = self.opt.lrC * (self.opt.lrCdecay ** tf.cast((iter // self.opt.lrCstep), tf.float32))
            decay = tf.cast(self.opt.lrGP / self.opt.epochs, tf.float32)
            lrGP = tf.cast(self.opt.lrGP * 1. / (1. + decay * epoch), tf.float32)
            lrC = tf.cast(self.opt.lrC * 1. / (1. + decay * epoch), tf.float32)
            print(f'epoch: {epoch}, lrGP: {lrGP}, lrC: {lrC}')
            self.op_IC_STN = tf.optimizers.SGD(lrGP, momentum=0.8)
            self.op_CNN = tf.optimizers.SGD(lrC, momentum=0.8)

            with self.train_writer.as_default():
                tf.summary.scalar('lrGP(epoch)', lrGP, step=epoch)
                tf.summary.scalar('lrC(epoch)', lrC, step=epoch)

            if (epoch + 1) % 10 == 0:
                self.ckpt_manager.save(epoch + 1)
                # ---------plot 前 testing 5筆 warp狀況---------
                Init_p = load_data.genPerturbations(self.opt)
                _, _, imageWarp, x_cor, y_cor = self.test_step(Init_p, self.testing_example)
                self.plot_Ori_region(self.testing_example, imageWarp, x_cor, y_cor, epoch, pad_width=8)

        # ---------testing---------
        A_te = []
        for image_batch, label_batch in self.test_datasets:
            Init_p = load_data.genPerturbations(self.opt)
            _, acc_te, _, _, _ = self.test_step(Init_p, image_batch, label_batch)
            A_te.append(acc_te)
        print(f'testing acc:{tf.reduce_mean(A_te).numpy()}')

if __name__ == '__main__':
    project = Project()
    project.run()