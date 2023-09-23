__all__ = ['get_nn_target_layer']


def get_nn_target_layer(model, model_name):
    target_layer = []
    if model_name == 'resnet':
        target_layer.append(model.features[-1])
    elif model_name == 'inception':
        target_layer.append(model.Mixed_7c)
    elif model_name == 'densenet':
        target_layer.append(model.features[-1])
    elif model_name == 'resnet-wam':
        target_layer.append(model.features_w[-1])
    elif model_name == 'resnet-wam-concat-t1':
        target_layer.append(model.features_w[-1])
    elif model_name == 'resnet-wam-alpha-t2':
        target_layer.append(model.layer4)
    elif model_name == 'resnet-wam-feature-t3':
        target_layer.append(model.feature_alpha)
    elif model_name == 'resnet-wam-concat-independence_net-t4':
        target_layer.append(model.features_x[-1])
    elif model_name == 'm3_n' or model_name == 'm3' or model_name == 'TransMUF':
        target_layer.append(model.module.features_x.ResNet_Module.layer4)
    elif 'transformer' in model_name:
        target_layer.append(model.transformer_model.norm)
    return target_layer
