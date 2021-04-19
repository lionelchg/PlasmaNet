import argparse
import yaml
import PlasmaNet.nnet.model as model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Config filename')
    parser.add_argument('-n', '--network', help='Network type (msnet or unet')
    args = parser.parse_args()

    with open(args.config) as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)
    
    for key, value in cfg.items():
        print(f'Architecture: {key}')
        if args.network == 'unet':
            value['args']['input_res'] = 101
            value['args']['up_type'] = 'upsample'
        else:
            value['args']['pad_method'] = 'zeros'
        tmp_model = getattr(model, value['type'])(**value['args'])
        print(tmp_model)
        print(f'Number of trainable parameters: {tmp_model.nparams:d}\n')
