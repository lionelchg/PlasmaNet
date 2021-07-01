import argparse
import yaml
import PlasmaNet.nnet.model as model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Config filename')
    parser.add_argument('-m', '--mode', help='Show all the network or just global parameters',
                            default='global')
    args = parser.parse_args()

    with open(args.config) as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)
    
    for key, value in cfg.items():
        print(f'Architecture: {key}')
        value['args']['input_res'] = 101
        tmp_model = getattr(model, value['type'])(**value['args'])
        if args.mode == 'all': 
            print(tmp_model)
        elif args.mode == 'global':
            print(tmp_model.global_prop())

if __name__ == '__main__':
    main()