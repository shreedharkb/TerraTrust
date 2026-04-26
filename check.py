import json
m = json.load(open('results/model_metrics.json'))
d = m['Model_D_Credit_Risk']['spatial']
print(f"D train Acc: {d.get('train_accuracy')} test Acc: {d.get('test_accuracy')}")
print(f"D precision: {d.get('precision')} recall: {d.get('recall')} f1: {d.get('f1_score')}")
