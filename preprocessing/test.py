from preprocessing import Traveser

code = open("test.java", "r").read()


def main():
    visitor = Traveser('java', code)
    visitor.travesal()


    # print(visitor.nodes)
    # print(visitor.control_flow_nodes)
    # print(visitor.edges)

main()