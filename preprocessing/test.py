from re import S
from preprocessing import Traveser, preprocess
import os

code = open("test.java", "r").read()


def main():
    visitor = Traveser('java', code)
    visitor.travesal()

    # print(visitor.nodes)
    # print(visitor.control_flow_nodes)
    ve = visitor.encode()
    print(ve)


def encode(set, type = 'fullGraph'):
    input_file = os.path.join(os.path.dirname(os.getcwd()), 'data/medium/%s/data.prev_full_code' % set)
    os.makedirs( os.path.join(os.path.dirname(os.getcwd()), 'encode-data/medium/%s' % set), exist_ok=True)
    out_file = os.path.join(os.path.dirname(os.getcwd()), ('encode-data/medium/%s/data.full_code_%s' % (set, type)))
    preprocess(input_file, out_file, type)

def move_commit(set): 
    input_file = os.path.join(os.path.dirname(os.getcwd()), 'data/medium/%s/data.commit_msg' % set)
    output_file = os.path.join(os.path.dirname(os.getcwd()), 'encode-data/medium/%s/data.commit_msg' % set)
    os.system(('cp %s %s' % (input_file, output_file)))

def move_buggy(set): 
    input_file = os.path.join(os.path.dirname(os.getcwd()), 'data/medium/%s/data.buggy_only' % set)
    output_file = os.path.join(os.path.dirname(os.getcwd()), 'encode-data/medium/%s/data.buggy_only' % set)
    os.system(('cp %s %s' % (input_file, output_file)))


if __name__ == '__main__':
    type = 'leaveOnly'
    for set in ['train', 'eval', 'test']:
        print('Encoding: ', set, '.....')
        encode(set, type)
        print('Copy Commit Msg: ', set, '.....')
        move_commit(set)
        print('Copy Buggy Code: ', set, '.....')
        move_buggy(set)
    

