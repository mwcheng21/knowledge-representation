from re import S
from preprocessing import Traveser, preprocess
import os

code = open("test.java", "r").read()


# def main():
#     visitor = Traveser('java', code)
#     visitor.travesal()

#     # print(visitor.nodes)
#     # print(visitor.control_flow_nodes)
#     print(visitor.patch_tokens.)
# main()


def encode(set, type = 'fullGraph'):
    input_file = os.path.join(os.path.dirname(os.getcwd()), 'original-data/small/%s/data.parent_full_code_hl' % set)
    os.makedirs( os.path.join(os.path.dirname(os.getcwd()), 'data/small/%s' % set), exist_ok=True)
    out_file = os.path.join(os.path.dirname(os.getcwd()), ('data/small/%s/data.full_code_%s' % (set, type)))
    out_buggy = os.path.join(os.path.dirname(os.getcwd()), ('data/small/%s/data.buggy_only_lw_%s' % (set, type)))
    preprocess(input_file, out_file, out_buggy, type)

def move_commit(set): 
    input_file = os.path.join(os.path.dirname(os.getcwd()), 'original-data/small/%s/data.commit_msg' % set)
    output_file = os.path.join(os.path.dirname(os.getcwd()), 'data/small/%s/data.commit_msg' % set)
    os.system(('cp %s %s' % (input_file, output_file)))

def move_buggy(set): 
    input_file = os.path.join(os.path.dirname(os.getcwd()), 'original-data/small/%s/data.parent_buggy_only' % set)
    output_file = os.path.join(os.path.dirname(os.getcwd()), 'data/small/%s/data.parent_buggy_only' % set)
    os.system(('cp %s %s' % (input_file, output_file)))

def move_fixed(set): 
    input_file = os.path.join(os.path.dirname(os.getcwd()), 'original-data/small/%s/data.child_code' % set)
    output_file = os.path.join(os.path.dirname(os.getcwd()), 'data/small/%s/data.fixed_only' % set)
    os.system(('cp %s %s' % (input_file, output_file)))


if __name__ == '__main__':
        # if type == 'fullGraph':
        #     nodes, edges = self.fullGraph()            
        # elif type == 'leaveOnly':
        #     nodes, edges = self.leaveOnly()        
        # elif type == 'astOnly':
        #     nodes, edges = self.astEdgeOnly()      
        # elif type == 'cfgOnly':
        #     nodes, edges = self.cfgEdgeOnly()    
        # elif type == 'dfgOnly':
        #     nodes, edges = self.dfgEdgeOnly()   
    type = 'fullGraph'
    for set in ['train', 'eval', 'test']:
        print('Encoding: ', set, '.....')
        #encode(set, type)
        print('Copy Commit Msg: ', set, '.....')
        move_commit(set)
        print('Copy Buggy Code: ', set, '.....')
        move_buggy(set)
        move_fixed(set)
    pass

