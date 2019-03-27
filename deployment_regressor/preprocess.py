import json
import sys
from trainer import regressor

if __name__=='__main__':
  df = regressor.get_data(sys.argv[1])
  sample = next(df.iterrows())

  input_sample = {}
  values = df.values[-24][0:193]
  print(values)
  input_sample['input'] = [values.tolist()]

  with open(sys.argv[2], 'w') as outfile:
    json.dump(input_sample, outfile)
