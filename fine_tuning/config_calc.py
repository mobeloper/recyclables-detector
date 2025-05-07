
# run: 
# python ./fine_tuning/config_calc.py

total_training_samples = 361
total_validation_samples = 45

train_batch_size = 20
val_batch_size = 10
num_epochs = 10
warmup_learning_rate = 0.0001
initial_learning_rate = 0.001

steps_per_loop = total_training_samples // train_batch_size
summary_interval = steps_per_loop
train_steps = num_epochs * steps_per_loop
validation_interval = steps_per_loop
validation_steps = total_validation_samples // val_batch_size
warmup_steps = steps_per_loop * 10
checkpoint_interval = steps_per_loop * 5
decay_steps = int(train_steps)

print(f'steps_per_loop: {steps_per_loop}')
print(f'summary_interval: {summary_interval}')
print(f'train_steps: {train_steps}')
print(f'validation_interval: {validation_interval}')
print(f'validation_steps: {validation_steps}')
print(f'warmup_steps: {warmup_steps}')
print(f'warmup_learning_rate: {warmup_learning_rate}')
print(f'initial_learning_rate: {initial_learning_rate}')
print(f'decay_steps: {decay_steps}')
print(f'checkpoint_interval: {checkpoint_interval}')

# steps_per_loop: 14
# summary_interval: 14
# train_steps: 700
# validation_interval: 14
# validation_steps: 9
# warmup_steps: 140
# warmup_learning_rate: 0.0001
# initial_learning_rate: 0.001
# decay_steps: 700
# checkpoint_interval: 70
