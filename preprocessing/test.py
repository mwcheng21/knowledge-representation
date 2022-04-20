from preprocessing import Traveser, preprocess
import os

code = open("test.java", "r").read()


def main():
    visitor = Traveser('java', code)
    visitor.travesal()

    # print(visitor.nodes)
    # print(visitor.control_flow_nodes)
    n, e = visitor.dfgEdgeOnly()
    print(n)
    print(e)


def main2():
    input_file = os.path.join(os.path.dirname(os.getcwd()), 'data/medium/train/data.buggy_only')
    os.makedirs( os.path.join(os.path.dirname(os.getcwd()), 'encoded-data/medium/train'), exist_ok=True)
    out_file = os.path.join(os.path.dirname(os.getcwd()), 'encoded-data/medium/train/data.buggy_only')
    preprocess(input_file, out_file, 'cfgOnly')


if __name__ == '__main__':
    main2()

