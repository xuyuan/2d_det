_base_ = './schedule_1x.py'
# learning policy
lr_config = dict(step=[16, 22])
runner = dict(max_epochs=24)
