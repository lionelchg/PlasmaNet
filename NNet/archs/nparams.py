import yaml
import PlasmaNet.nnet.model as model

if __name__ == '__main__':
    with open('unets.yml') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)
    
    for key, value in cfg.items():
        print(f'Architecture: {key}')
        value['args']['input_res'] = 101
        tmp_model = getattr(model, value['type'])(**value['args'])
        print(tmp_model)