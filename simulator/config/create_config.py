import json

js = json.loads(open("/home/ma-user/modelarts/Nuclear/config/ITER_cocos02.json").read())

print(len(js['core_profiles']['profiles_1d'][0]['electrons']['density']))
print(js['core_profiles']['profiles_1d'][0]['electrons']['density'][:10])
js['core_profiles']['profiles_1d'][0]['electrons']['density'] = [1e20 for i in range(201)]

json.dump(js, open("/home/ma-user/modelarts/Nuclear/config/running_config.json", "w"))
