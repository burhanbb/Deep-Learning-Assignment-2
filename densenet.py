import math, re, os, time
import tensorflow as tf, tensorflow.keras.backend as K
import numpy as np
from collections import namedtuple
from matplotlib import pyplot as plt
from kaggle_datasets import KaggleDatasets
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tensorflow.keras.applications import DenseNet201

print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

GCS_DS_PATH = KaggleDatasets().get_gcs_path('flower-classification-with-tpus')

IMAGE_SIZE = [192, 192]
EPOCHS = 12
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
AUG_BATCH = BATCH_SIZE

GCS_PATH_SELECT = {
    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192'
}
GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')
VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec')

CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49
           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']

np.set_printoptions(threshold=15, linewidth=80)

def batch_to_numpy_images_and_labels(data):
    images, labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    if numpy_labels.dtype == object:
        numpy_labels = [None for _ in enumerate(numpy_images)]
    return numpy_images, numpy_labels

def title_from_label_and_target(label, correct_label):
    if correct_label is None:
        return CLASSES[label], True
    correct = (label == correct_label)
    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',
                                CLASSES[correct_label] if not correct else ''), correct

def display_one_flower(image, title, subplot, red=False, titlesize=16):
    plt.subplot(*subplot)
    plt.axis('off')
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)
    
def display_batch_of_images(databatch, predictions=None):
    images, labels = batch_to_numpy_images_and_labels(databatch)
    if labels is None:
        labels = [None for _ in enumerate(images)]
    rows = int(math.sqrt(len(images)))
    cols = len(images)//rows
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols,1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))
    
    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):
        if label is None:
            title = ''
        elif isinstance(label, int):
            title = CLASSES[label]
        else:
            idx = np.argmax(label, axis=0)
            title = CLASSES[idx]
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3
        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)
    
    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()

def display_confusion_matrix(cmat, score, precision, recall):
    plt.figure(figsize=(15,15))
    ax = plt.gca()
    ax.matshow(cmat, cmap='Reds')
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    titlestring = ""
    if score is not None:
        titlestring += 'f1 = {:.3f} '.format(score)
    if precision is not None:
        titlestring += '\nprecision = {:.3f} '.format(precision)
    if recall is not None:
        titlestring += '\nrecall = {:.3f} '.format(recall)
    if len(titlestring) > 0:
        ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})
    plt.show()
    
def display_training_curves(training, validation, title, subplot):
    if subplot%10==1:
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])

def get_batch_transformatioin_matrix(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    batch_size = tf.cast(tf.reduce_sum(tf.ones_like(rotation)), tf.int64)
    rotation = tf.constant(math.pi) * rotation / 180.0
    shear = tf.constant(math.pi) * shear / 180.0
    one = tf.ones_like(rotation, dtype=tf.float32)
    zero = tf.zeros_like(rotation, dtype=tf.float32)
    
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)

    rotation_matrix_temp = tf.stack([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0)
    rotation_matrix_temp = tf.transpose(rotation_matrix_temp)
    rotation_matrix = tf.reshape(rotation_matrix_temp, shape=(batch_size, 3, 3))
        
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    
    shear_matrix_temp = tf.stack([one, s2, zero, zero, c2, zero, zero, zero, one], axis=0)
    shear_matrix_temp = tf.transpose(shear_matrix_temp)
    shear_matrix = tf.reshape(shear_matrix_temp, shape=(batch_size, 3, 3))    
    
    zoom_matrix_temp = tf.stack([one / height_zoom, zero, zero, zero, one / width_zoom, zero, zero, zero, one], axis=0)
    zoom_matrix_temp = tf.transpose(zoom_matrix_temp)
    zoom_matrix = tf.reshape(zoom_matrix_temp, shape=(batch_size, 3, 3))
    
    shift_matrix_temp = tf.stack([one, zero, height_shift, zero, one, width_shift, zero, zero, one], axis=0)
    shift_matrix_temp = tf.transpose(shift_matrix_temp)
    shift_matrix = tf.reshape(shift_matrix_temp, shape=(batch_size, 3, 3))    
        
    return tf.linalg.matmul(tf.linalg.matmul(rotation_matrix, shear_matrix), tf.linalg.matmul(zoom_matrix, shift_matrix))


def batch_transform(images, labels):
    DIM = images.shape[1]
    XDIM = DIM % 2
    
    batch_size = tf.cast(tf.reduce_sum(tf.ones_like(images)) / (images.shape[1] * images.shape[2] * images.shape[3]), tf.int64)
    
    rot = 15.0 * tf.random.normal([batch_size], dtype='float32')
    shr = 5.0 * tf.random.normal([batch_size], dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([batch_size], dtype='float32') / 10.0
    w_zoom = 1.0 + tf.random.normal([batch_size], dtype='float32') / 10.0
    h_shift = 16.0 * tf.random.normal([batch_size], dtype='float32') 
    w_shift = 16.0 * tf.random.normal([batch_size], dtype='float32') 
  
    m = get_batch_transformatioin_matrix(rot, shr, h_zoom, w_zoom, h_shift, w_shift) 

    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype='int32')
    idx = tf.stack([x, y, z])
    
    idx2 = tf.linalg.matmul(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)
    idx3 = tf.stack([DIM // 2 - idx2[:, 0, ], DIM // 2 - 1 + idx2[:, 1, ]], axis=1)  
    
    d = tf.gather_nd(images, tf.transpose(idx3, perm=[0, 2, 1]), batch_dims=1)
        
    new_images = tf.reshape(d, (batch_size, DIM, DIM, 3))

    return new_images, labels

def onehot(image,label):
    return image,tf.one_hot(label,len(CLASSES))

def batch_cutmix(images, labels, PROBABILITY=1.0, batch_size=0):
    
    DIM = IMAGE_SIZE[0]
    CLASSES = 104
    
    if batch_size == 0:
        batch_size = AUG_BATCH
    
    do_cutmix = tf.cast(tf.random.uniform([batch_size], 0, 1) <= PROBABILITY, tf.int32)
    new_image_indices = tf.cast(tf.random.uniform([batch_size], 0, batch_size), tf.int32)
    new_x = tf.cast(tf.random.uniform([batch_size], 0, DIM), tf.int32)
    new_y = tf.cast(tf.random.uniform([batch_size], 0, DIM), tf.int32)
    b = tf.random.uniform([batch_size], 0, 1)
    new_width = tf.cast(DIM * tf.math.sqrt(1-b), tf.int32) * do_cutmix
    
    new_y0 = tf.math.maximum(0, new_y - new_width // 2)
    new_y1 = tf.math.minimum(DIM, new_y + new_width // 2)
    new_x0 = tf.math.maximum(0, new_x - new_width // 2)
    new_x1 = tf.math.minimum(DIM, new_x + new_width // 2)
    target = tf.broadcast_to(tf.range(DIM), shape=(batch_size, DIM))
    mask_y = tf.math.logical_and(new_y0[:, tf.newaxis] <= target, target <= new_y1[:, tf.newaxis])
    mask_x = tf.math.logical_and(new_x0[:, tf.newaxis] <= target, target <= new_x1[:, tf.newaxis])
    mask = tf.cast(tf.math.logical_and(mask_y[:, :, tf.newaxis], mask_x[:, tf.newaxis, :]), tf.float32)

    new_images =  images * tf.broadcast_to(1 - mask[:, :, :, tf.newaxis], [batch_size, DIM, DIM, 3]) + \
                    tf.gather(images, new_image_indices) * tf.broadcast_to(mask[:, :, :, tf.newaxis], [batch_size, DIM, DIM, 3])

    a = tf.cast(new_width ** 2 / DIM ** 2, tf.float32)    
        
    if len(labels.shape) == 1:
        labels = tf.one_hot(labels, CLASSES)
        
    new_labels =  (1-a)[:, tf.newaxis] * labels + a[:, tf.newaxis] * tf.gather(labels, new_image_indices)        
        
    return new_images, new_labels

def batch_mixup(images, labels, PROBABILITY=1.0, batch_size=0):

    DIM = IMAGE_SIZE[0]
    CLASSES = 104
    
    if batch_size == 0:
        batch_size = AUG_BATCH
    
    do_mixup = tf.cast(tf.random.uniform([batch_size], 0, 1) <= PROBABILITY, tf.int32)
    new_image_indices = tf.cast(tf.random.uniform([batch_size], 0, batch_size), tf.int32)
    a = tf.random.uniform([batch_size], 0, 1) * tf.cast(do_mixup, tf.float32)
    new_images =  (1-a)[:, tf.newaxis, tf.newaxis, tf.newaxis] * images + a[:, tf.newaxis, tf.newaxis, tf.newaxis] * tf.gather(images, new_image_indices)

    if len(labels.shape) == 1:
        labels = tf.one_hot(labels, CLASSES)
    new_labels =  (1-a)[:, tf.newaxis] * labels + a[:, tf.newaxis] * tf.gather(labels, new_image_indices)

    return new_images, new_labels

def transform_cut_mix(image,label):
    DIM = IMAGE_SIZE[0]
    CLASSES = 104
    SWITCH = 0.5
    CUTMIX_PROB = 0.666
    MIXUP_PROB = 0.666
    image2, label2 = batch_cutmix(image, label, CUTMIX_PROB)
    image3, label3 = batch_mixup(image, label, MIXUP_PROB)
    imgs = []; labs = []
    for j in range(AUG_BATCH):
        P = tf.cast( tf.random.uniform([],0,1)<=SWITCH, tf.float32)
        imgs.append(P*image2[j,]+(1-P)*image3[j,])
        labs.append(P*label2[j,]+(1-P)*label3[j,])
    image4 = tf.reshape(tf.stack(imgs),(AUG_BATCH,DIM,DIM,3))
    label4 = tf.reshape(tf.stack(labs),(AUG_BATCH,CLASSES))
    return image4,label4

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "class": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "id": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['id']
    return image, idnum

def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    return dataset

def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label   

def get_training_dataset(dataset, simple_aug=False, advance_aug=False, cut_mix_aug=0, do_onehot=True):
    if simple_aug:
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat()
    dataset = dataset.batch(AUG_BATCH)
    if advance_aug:
        dataset = dataset.map(batch_transform, num_parallel_calls=AUTO)
    if do_onehot:
        dataset = dataset.map(onehot, num_parallel_calls=AUTO)
    if cut_mix_aug==1:
        dataset = dataset.map(batch_cutmix, num_parallel_calls=AUTO)
    elif cut_mix_aug==2:
        dataset = dataset.map(batch_mixup, num_parallel_calls=AUTO)
    elif cut_mix_aug==3:
        dataset = dataset.map(transform_cut_mix, num_parallel_calls=AUTO)
    dataset = dataset.unbatch()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_validation_dataset(dataset, ordered=False, repeated=False, do_onehot=True):
    if repeated:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(2048)
    if do_onehot:
        dataset = dataset.map(onehot, num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=repeated)
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


def int_div_round_up(a, b):
    return (a + b - 1) // b

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
VALIDATION_STEPS = int_div_round_up(NUM_VALIDATION_IMAGES, BATCH_SIZE)
print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))

train_ds = load_dataset(TRAINING_FILENAMES, labeled=True)
valid_ds = load_dataset(VALIDATION_FILENAMES, labeled=True)
test_ds = load_dataset(TEST_FILENAMES, labeled=False)

print("Training data shapes:")
for image, label in get_training_dataset(train_ds).take(3):
    print(image.numpy().shape, label.numpy().shape)
print("Training data label examples:", label.numpy())

print("Validation data shapes:")
for image, label in get_validation_dataset(valid_ds).take(3):
    print(image.numpy().shape, label.numpy().shape)
print("Validation data label examples:", label.numpy())

print("Test data shapes:")
for image, idnum in get_test_dataset(test_ds).take(3):
    print(image.numpy().shape, idnum.numpy().shape)
print("Test data IDs:", idnum.numpy().astype('U'))


training_dataset = get_training_dataset(train_ds, simple_aug=True, advance_aug=False, cut_mix_aug=1)
training_dataset = training_dataset.unbatch().batch(20)
train_batch = iter(training_dataset)

display_batch_of_images(next(train_batch))

test_dataset = get_test_dataset(test_ds)
test_dataset = test_dataset.unbatch().batch(20)
test_batch = iter(test_dataset)

display_batch_of_images(next(test_batch))

images, labels = next(iter(training_dataset.take(1)))
new_images, new_labels = batch_cutmix(images, labels, PROBABILITY=1.0, batch_size=20)
display_batch_of_images((new_images, new_labels))

images, labels = next(iter(training_dataset.take(1)))
new_images, new_labels = batch_mixup(images, labels, PROBABILITY=1.0, batch_size=20)
display_batch_of_images((new_images, new_labels))

LR_START = 0.00003
LR_MAX = 0.00005 * strategy.num_replicas_in_sync
LR_MIN = 0.000001
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 3
LR_EXP_DECAY = .8
        
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

def lrfn(epoch):
    LR_START = 0.00001
    LR_MAX = 0.00005 * strategy.num_replicas_in_sync
    LR_MIN = 0.00001
    LR_RAMPUP_EPOCHS = 5
    LR_SUSTAIN_EPOCHS = 0
    LR_EXP_DECAY = .8
    
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

callbacks = [earlystop, lr_schedule]

rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, [lrfn(x) for x in rng])
print(y[0], y[-1])

with strategy.scope():
    pretrained_model = DenseNet201(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
    pretrained_model.trainable = True
    
    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
        
    class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __call__(self, step):
            return lrfn(epoch=step//STEPS_PER_EPOCH)

    model.compile(
        optimizer='adam',
        loss = 'categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    model.summary()
        
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    valid_accuracy = tf.keras.metrics.CategoricalAccuracy()
    train_loss = tf.keras.metrics.Sum()
    valid_loss = tf.keras.metrics.Sum()
    
    loss_fn = lambda a,b: tf.nn.compute_average_loss(tf.keras.losses.categorical_crossentropy(a,b), global_batch_size=BATCH_SIZE)

history = model.fit(
    get_training_dataset(train_ds, simple_aug=True, advance_aug=False, cut_mix_aug=3), 
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=get_validation_dataset(valid_ds), callbacks=[callbacks]
)

start_time = epoch_start_time = time.time()
simple_ctl_training_time = time.time() - start_time
print("OPTIMIZED CTL TRAINING TIME: {:0.1f}s".format(simple_ctl_training_time))

model.save_weights('densenet-flower-tpu.hdf5')

display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)
display_training_curves(history.history['categorical_accuracy'], history.history['val_categorical_accuracy'], 'accuracy', 212)

cmdataset = get_validation_dataset(load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=True), ordered=True, do_onehot=False)
images_ds = cmdataset.map(lambda image, label: image)
labels_ds = cmdataset.map(lambda image, label: label).unbatch()
cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy()
cm_probabilities = model.predict(images_ds)
cm_predictions = np.argmax(cm_probabilities, axis=-1)
print("Correct   labels: ", cm_correct_labels.shape, cm_correct_labels)
print("Predicted labels: ", cm_predictions.shape, cm_predictions)

cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))
score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
cmat = (cmat.T / cmat.sum(axis=1)).T
display_confusion_matrix(cmat, score, precision, recall)
print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))

test_ds = get_test_dataset(ordered=True)

print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds)
predictions = np.argmax(probabilities, axis=-1)
print(predictions)

accuracy = tf.keras.metrics.Accuracy()
accuracy.update_state(tf.argmax(predictions, axis=1), range(len(CLASSES)))
result = accuracy.result().numpy()

print(f"Accuracy: {result}")

dataset = get_validation_dataset(valid_ds, do_onehot=False)
dataset = dataset.unbatch().batch(20)
batch = iter(dataset)

images, labels = next(batch)
probabilities = model.predict(images)
predictions = np.argmax(probabilities, axis=-1)
display_batch_of_images((images, labels), predictions)