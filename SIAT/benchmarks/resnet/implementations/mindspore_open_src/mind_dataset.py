# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


"""
create train or eval dataset.
"""
import os
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.transforms.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.transforms.vision.py_transforms as P
from PIL import Image
from io import BytesIO

def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32

    Returns:
        dataset
    """
    device_num = int(os.getenv("RANK_SIZE"))
    try:
        # global_rank_id = int(os.getenv('RANK_ID').split("-")[1].split("custom")[1])
        global_rank_id = int(os.getenv('RANK_ID').split("-")[-1])
    except:
        global_rank_id = 0
    rank_id = int(os.getenv('DEVICE_ID')) + global_rank_id * 8

    columns_list = ["data","label"]
    if do_train:
        ds = de.MindDataset(dataset_path+'/imagenet_train.mindrecord00', columns_list,  num_parallel_workers=8, shuffle=True, 
                            num_shards=device_num, shard_id=rank_id)
        print("train dataset size", ds.get_dataset_size())
    else:
        padded_sample = {}
        white_io = BytesIO()
        Image.new('RGB',(224,224),(255,255,255)).save(white_io, 'JPEG')
        padded_sample['data'] = white_io.getvalue()
        padded_sample['label'] = -1
        batch_per_step = batch_size * device_num
        print("eval batch per step:", batch_per_step)
        if batch_per_step < 50000:
            if 50000 % batch_per_step == 0:
                num_padded = 0
            else:
                num_padded = batch_per_step - (50000 % batch_per_step)
        else:
            num_padded = batch_per_step - 50000
        print("Padded samples:", num_padded)
        ds = de.MindDataset(dataset_path+'/imagenet_eval.mindrecord0', columns_list, num_parallel_workers=8, shuffle=False, 
                            num_shards=device_num, shard_id=rank_id, padded_sample=padded_sample, num_padded=num_padded)
        print("eval dataset size", ds.get_dataset_size())


    image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            C.RandomHorizontalFlip(prob=0.5),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]
    else:
        trans = [
            C.Decode(),
            C.Resize((365, 365)),
            C.CenterCrop(320),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(mstype.int32)

    # ds = ds.shuffle(buffer_size=100000)
    ds = ds.map(input_columns="data", num_parallel_workers=8, operations=trans)
    ds = ds.map(input_columns="label", num_parallel_workers=8, operations=type_cast_op)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)

    return ds

