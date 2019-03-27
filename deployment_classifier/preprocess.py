import json
import sys
from trainer import classifier

if __name__=='__main__':
  df, weights = classifier.get_data('../../../data.csv')
  sample = next(df.iterrows())

  input_sample = {}
  values = df.values[-24][0:193]
  print(values)
  input_sample['input'] = [values.tolist()]

  with open(sys.argv[1], 'w') as outfile:
    json.dump(input_sample, outfile)
