"""
This module creates cyclical learning rate values and boundaries for our 
jenkins-based infrastructure for deep learning experiments. 
It produces two lists that should be pasted to the relevant textboxes in jenkins
"""
# todo @ilan: add to jenkins as GUI
from __future__ import print_function

# Change these values
step_size = 500
max_lr = 1e-8
min_lr = 1e-9
images_per_batch = 10

# Don't change these values
boundaries = [x for x in range(1, images_per_batch * step_size)]
values_up = [min_lr + (max_lr - min_lr) / step_size * x for x in
             range(0, step_size)]
values_down = [max_lr - (max_lr - min_lr) / step_size * x for x in
               range(0, step_size)]
values_final = values_up + values_down

print('Copy these lists to the jenkins relevant textboxes to control LR')
print('Values:')
print(values_final)
print('Boundaries:')
print(boundaries)
