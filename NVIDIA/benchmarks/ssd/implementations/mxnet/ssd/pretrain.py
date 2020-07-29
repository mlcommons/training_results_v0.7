import pickle
import numpy as np

SSD_LAYERS = ['expand_trans_conv', 'expand_conv'] # Layers that are not part of the backbone

def pretrain_backbone(param_dict,
                      picklefile_name,
                      layout='NCHW',
                      backbone_prefix='ssd0_resnetmlperf0_'):
    with open(picklefile_name, 'rb') as picklefile:
        pretrained_dict = pickle.load(picklefile)

    for param_name in param_dict.keys():
        # Skip layers not part of the backbone
        if any(n in param_name for n in SSD_LAYERS):
            continue

        # convert parameter name to match the names in the pretrained file
        pretrained_param_name = param_name
        # Remove backbone_prefix from name
        pretrained_param_name = pretrained_param_name.replace(backbone_prefix, '')
        # 'batchnormaddrelu' uses 'moving' rather than 'running' for mean/var
        pretrained_param_name = pretrained_param_name.replace('moving', 'running')

        assert pretrained_param_name in pretrained_dict, \
               f'Can\'t find parameter {pretrained_param_name} in the picklefile'
        param_type = type(pretrained_dict[pretrained_param_name])
        assert isinstance(pretrained_dict[pretrained_param_name], np.ndarray), \
               f'Parameter {pretrained_param_name} in the picklefile has a wrong type ({param_type})'

        pretrained_weights = pretrained_dict[pretrained_param_name]

        if layout == 'NHWC' and pretrained_weights.ndim==4:
            # Place channels into last dim
            pretrained_weights = pretrained_weights.transpose((0, 2, 3, 1))

            # this special case is intended only for the first
            # layer, where the channel count needs to be padded
            # from 3 to 4 for NHWC
            if (pretrained_weights.shape[3]+1)==param_dict[param_name].shape[3]:
                pretrained_weights = np.pad(pretrained_weights,
                                            ((0, 0), (0, 0), (0, 0), (0, 1)),
                                            mode='constant')

        assert param_dict[param_name].shape == pretrained_weights.shape, \
               'Network parameter {} and pretrained parameter {} have different shapes ({} vs {})' \
               .format(param_name, pretrained_param_name, param_dict[param_name].shape, pretrained_weights.shape)
        param_dict[param_name].set_data(pretrained_weights)
