from aifactory.modules import activate, submit

activate('wildgrapes18@gmail.com', 'qkrxo6264!')

task = 10
code = r"/tf/notebooks/10_covid/train.py"
weight = r"/tf/notebooks/10_covid/pth/10_covid_4_0.81_0.10653.pth"
result = r"/tf/notebooks/10_covid/prediction.txt"
submit(task, code, weight, result)