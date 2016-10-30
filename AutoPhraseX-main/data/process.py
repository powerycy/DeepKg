import re
import json
def read_json(file):
    with open(file) as f:
        train = json.load(f)
        with open('医疗_vocab.txt','a',encoding='utf8') as f1:
            for line in train['nodes']:
                # line = json.loads(line)
                text = line['name']
                f1.write(text+'\n')
def read_rel_json(file):
    with open(file) as f:
        train = json.load(f)
        with open('医疗_vocab.txt','a',encoding='utf8') as f1:
            for _,value in train['_default'].items():
                sub = value['subject']
                f1.write(sub + '\n')
                obj = value['object']
                f1.write(obj + '\n')
                rel = value['relation']
                f1.write(rel + '\n')
def merge_file(file):
    set_file = set()
    with open(file, 'r', encoding='utf8') as f2:
        with open('医疗_vocab.txt', 'a', encoding='utf8') as f1:
            for line in f2:
                set_file.add(line)
            for line in set_file:
                f1.write(line)


if __name__ == '__main__':
    # read_rel_json('/Users/xm20201013/Documents/20201214/symptom.json')
    merge_file('/Users/xm20201013/Documents/AutoPhraseX-main/data/医疗_vocab.txt')
