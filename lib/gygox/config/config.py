import yaml
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
config_yaml_path = os.path.join(current_dir, 'config.yaml')

with open(config_yaml_path, 'r') as stream:
    cfg = yaml.load(stream)

# gygo_project = os.path.join('projects/gygo')
# cfg['paths']['gygo_project'] = os.path.join(
#     cfg['paths']['caffe_root'], 'projects', 'gygo')
